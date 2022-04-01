import argparse   # argparse 是python自带的命令行参数解析包
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader # 数据加载
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True) # 用于递归创建目录

# 创建解析器
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")  # 使用的adam 学习速率 设置的比较小
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")   # 优化器adam 的两个参数 矩估计的指数衰减率
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")  # 使用cpu 的数量
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") # 随机噪声z的维度
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension") # 输入图像的尺寸
parser.add_argument("--channels", type=int, default=1, help="number of image channels")  # 输入图像的channel数 1是灰度图像  3是RGB
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples") # 保存生成模型图像的间隔
opt = parser.parse_args() # 所有参数输出
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)  # 图像尺寸 (1.28.28)
print('img_shape=',img_shape)

cuda = True if torch.cuda.is_available() else False # 使用cuda

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.Resize(opt.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
        ),#对图像进行transform。有重新设置尺寸到我们所需要的大小，转变成tensor，归一化
    ),

    batch_size=opt.batch_size,
    shuffle=True,#将加载好的数据打乱
)

# %%

import matplotlib.pyplot as plt


def show_img(img, trans=True):
    if trans:
        img = np.transpose(img.detach().cpu().numpy(), (1, 2, 0))  # 把channel维度放到最后
        plt.imshow(img[:, :, 0], cmap="gray")
    else:
        plt.imshow(img, cmap="gray")
    plt.show()


mnist = datasets.MNIST("../../data/mnist")

for i in range(3):#把mnist中的前三张图片展示出来
    sample = mnist[i][0]
    label = mnist[i][1]
    show_img(np.array(sample), trans=False)
    print("label =", label, '\n')

#为了展现加载数据的transform过程
trans_resize = transforms.Resize(opt.img_size)
trans_to_tensor = transforms.ToTensor()
trans_normalize = transforms.Normalize([0.5], [0.5]) # x_n = (x - 0.5) / 0.5归一化的过程

print("shape =", np.array(sample).shape, '\n')
print("data =", np.array(sample), '\n')
samlpe = trans_resize(sample)
print("(trans_resize) shape =", np.array(sample).shape, '\n')
sample = trans_to_tensor(sample)
print("(trans_to_tensor) data =", sample, '\n')
sample = trans_normalize(sample)#归一化后的
print("(trans_normalize) data =", sample, '\n')

# %%
class Generator(nn.Module): # 生成器 5个全连接层
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]  # Linear全连接层 输入维度，输出维度
            if normalize:  # 要不要正则化
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # LeakyReLU的激活函数，，小于0ai=0.2
            return layers

        self.model = nn.Sequential(
            *block(opt.latent_dim, 128, normalize=False), # 调整维度，不正则化
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),  # 这三正则化
            nn.Linear(1024, int(np.prod(img_shape))),  # 全连接层 24转化为784维度
            nn.Tanh()  # 值域转化为（-1，1）
        )

    def forward(self, z):  # 把随机噪声z输入到定义的模型
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

generator = Generator()#把generator进行实例化，得到了一个模型的实例
print(generator)

class Discriminator(nn.Module): # 判别器
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),  # 把值域变化到 0-1
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

discriminator = Discriminator()
print(discriminator)


# Loss function 损失函数
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()  # 实例化
discriminator = Discriminator()

# cuda 加速
if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()



# Optimizers 优化器Adam
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))  # 使用参数的学习速率，只优化生成器中的参数
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))#只优化辨别器中的参数

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):   # 从数据集和随机向量仲获取输入
        # Configure input
        real_imgs = Variable(imgs.type(Tensor))  # 用来保存真实值
        # Sample noise as generator input
        z = Variable(Tensor( np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))  # 定义随机噪声，从正态分布均匀采样均值为0，方差为1  opt.latent_dim = 100
        print("i =", i, '\n')
        print("shape of z =", z.shape, '\n')
        print("shape of real_imgs =", real_imgs.shape, '\n')
        print("z =", z, '\n')
        print("real_imgs =")

        for img in real_imgs[:3]:
            show_img(img)#这里随机展示的数据与上面展示的3个数据是不一样的，因为我们在加载数据的过程中已经将顺序打乱



        # 分别计算loss，使用反向传播更新模型
        # Adversarial ground truths t是标注.正确的t标注是ground truth
        valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)  # 判定1为真
        fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)  # 判定0为假



        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()  # 把生成器的梯度清0



        # Generate a batch of images
        gen_imgs = generator(z)  # 生成图像
        print("gen_imgs =")
        for img in gen_imgs[:3]:
            show_img(img)

        # Loss measures generator's ability to fool the discriminator
        #把生成的图像送入判别器，得到判别器 对他的输出
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)   # 对抗loss 最小化需要判别器的输出和真（1）尽量接近

        g_loss.backward()  # 反向传播
        optimizer_G.step()  # 模型更新

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)#梯度到了gen_imgs就不会再往前传避免了一些多余的计算

        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        ## 9 保存生成图像和模型文件

        # %%



        epoch = 0  # temporary

        batches_done = epoch * len(dataloader) + i  # 算一个总的batch
        if batches_done % opt.sample_interval == 0:   # 当batch数等于设定参数的倍数的时候
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)  # 保存图像

            os.makedirs("model", exist_ok=True)  # 保存模型
            torch.save(generator, 'model/generator.pkl')
            torch.save(discriminator, 'model/discriminator.pkl')
            print("gen images saved!\n")
            print("model saved!")

        # %%


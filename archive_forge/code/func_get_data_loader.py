import ray
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
from torch.autograd import Variable
from torch.nn import functional as F
from scipy.stats import entropy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
def get_data_loader(data_dir='~/data'):
    dataset = dset.MNIST(root=data_dir, download=True, transform=transforms.Compose([transforms.Resize(image_size), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    return dataloader
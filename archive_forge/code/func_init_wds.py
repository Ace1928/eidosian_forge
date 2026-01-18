import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime
def init_wds(self, bias=0):

    def identity(x):
        return x
    import webdataset as wds
    import torchvision.transforms as transforms
    img_transform = transforms.Compose([transforms.CenterCrop(512), transforms.Resize(args.my_img_size)])
    self.data_raw = wds.WebDataset(args.data_file, resampled=True).shuffle(10000, initial=1000, rng=random.Random(epoch * 100000 + rank + bias * 1000000000.0)).decode('torchrgb').to_tuple('jpg', 'json', 'txt').map_tuple(img_transform, identity, identity)
    for pp in self.data_raw.pipeline:
        if 'Resampled' in str(pp):
            pp.deterministic = True

            def worker_seed():
                return rank * 100000 + epoch + bias * 1000000000.0
            pp.worker_seed = worker_seed
    self.data = iter(self.data_raw)
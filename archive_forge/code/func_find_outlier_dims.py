import json
import shlex
import subprocess
from typing import Tuple
import torch
def find_outlier_dims(weight, reduction_dim=0, zscore=4.0, topk=None, rdm=False):
    if rdm:
        return torch.randint(0, weight.shape[1], size=(topk,), device=weight.device).long()
    m = weight.mean(reduction_dim)
    mm = m.mean()
    mstd = m.std()
    zm = (m - mm) / mstd
    std = weight.std(reduction_dim)
    stdm = std.mean()
    stdstd = std.std()
    zstd = (std - stdm) / stdstd
    if topk is not None:
        val, idx = torch.topk(std.abs(), k=topk, dim=0)
    else:
        idx = torch.where(zstd > zscore)[0]
    return idx
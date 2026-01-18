import functools
import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.cpp_extension import load
@MyFunction
def jit_func(self, x):
    B, T, C = x.size()
    xx = self.time_shift(x)
    xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)
    r = self.receptance(xr)
    k = self.key(xk)
    v = self.value(xv)
    g = F.silu(self.gate(xg))
    return (r, k, v, g)
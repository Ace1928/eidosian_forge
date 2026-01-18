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
def jit_funcQKV(self, x):
    xx = self.time_shift(x)
    xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
    xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
    xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
    xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
    xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
    xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)
    k = self.key(xk)
    v = self.value(xv)
    r = self.receptance(xr)
    sr = torch.sigmoid(r)
    qq = self.qq(xqq)
    kk = self.kk(xkk)
    vv = self.vv(xvv)
    return (sr, k, v, qq, kk, vv)
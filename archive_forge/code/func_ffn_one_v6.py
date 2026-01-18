import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
from torch import nn
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
from typing import Optional
import types, gc, os, time, re
import torch
import torch.nn as nn
from torch.nn import functional as F
@MyFunction
def ffn_one_v6(self, x, sx, ln_w, ln_b, k_maa, r_maa, kw, vw, rw, kmx, krx, kmy, kry, vmx, vrx, vmy, vry, rmx, rrx, rmy, rry):
    xx = F.layer_norm(x, (x.shape[-1],), weight=ln_w, bias=ln_b)
    sx = sx - xx
    kx = xx + sx * k_maa
    rx = xx + sx * r_maa
    r = torch.sigmoid(matmul(rx, rw, rmx, rrx, rmy, rry))
    vx = torch.relu(matmul(kx, kw, kmx, krx, kmy, kry)) ** 2
    out = r * matmul(vx, vw, vmx, vrx, vmy, vry)
    return (x + out, xx)
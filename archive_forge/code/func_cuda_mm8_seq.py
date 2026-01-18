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
@MyStatic
def cuda_mm8_seq(B: int, N: int, M: int, x, w, mx, rx, my, ry):
    assert x.dtype == mx.dtype == rx.dtype == my.dtype == ry.dtype
    assert x.dtype == torch.float32 or x.dtype == torch.float16
    assert w.dtype == torch.uint8
    assert x.shape == (B, N)
    assert w.shape == (N, M)
    assert rx.shape == mx.shape == (M,)
    assert ry.shape == my.shape == (N, 1)
    y = torch.empty((B, M), device=w.device, dtype=x.dtype)
    torch.ops.rwkv.mm8_seq(B, N, M, x, w, mx, rx, my, ry, y)
    return y
import copy
from typing import Any, Dict, Optional, TypeVar, Union, overload
import warnings
import torch
from torch import Tensor, device, dtype, nn
import torch.nn.functional as F
import bitsandbytes as bnb
from bitsandbytes.autograd._functions import get_tile_inds, undo_layout
from bitsandbytes.functional import QuantState
from bitsandbytes.optim import GlobalOptimManager
from bitsandbytes.utils import OutlierTracer
def set_compute_type(self, x):
    if x.dtype in [torch.float32, torch.bfloat16]:
        self.compute_dtype = x.dtype
    elif x.dtype == torch.float16:
        if self.compute_dtype == torch.float32 and x.numel() == x.shape[-1]:
            warnings.warn('Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference.')
            warnings.filterwarnings('ignore', message='.*inference.')
        if self.compute_dtype == torch.float32 and x.numel() != x.shape[-1]:
            warnings.warn('Input type into Linear4bit is torch.float16, but bnb_4bit_compute_dtype=torch.float32 (default). This will lead to slow inference or training speed.')
            warnings.filterwarnings('ignore', message='.*inference or training')
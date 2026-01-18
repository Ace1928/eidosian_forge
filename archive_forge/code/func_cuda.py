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
def cuda(self, device):
    if self.has_fp16_weights:
        return super().cuda(device)
    else:
        B = self.data.contiguous().half().cuda(device)
        CB, CBt, SCB, SCBt, coo_tensorB = bnb.functional.double_quant(B)
        del CBt
        del SCBt
        self.data = CB
        self.CB = CB
        self.SCB = SCB
    return self
import math
from enum import Enum
from functools import partial
from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch._prims_common as utils
from torch import SymBool, SymFloat, Tensor
from torch._decomp import (
from torch._ops import OpOverload
from torch._prims import _prim_elementwise_meta, ELEMENTWISE_PRIM_TYPE_PROMOTION_KIND
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _broadcast_shapes, _maybe_broadcast
from torch.utils import _pytree as pytree
import torch._refs
import torch._refs.nn.functional
import torch._refs.special
@register_meta(torch.ops.quantized.max_pool2d)
def meta_quantized_max_pool2d(input, kernel_size, stride=(), padding=(0,), dilation=(1,), ceil_mode=False):
    nInputPlane, outputHeight, outputWidth = max_pool2d_checks_and_compute_shape(input, kernel_size, stride, padding, dilation, ceil_mode)
    nbatch = input.size(-4) if input.dim() == 4 else 1
    memory_format = torch.channels_last
    if input.dim() == 3:
        size = [nInputPlane, outputHeight, outputWidth]
    else:
        size = [nbatch, nInputPlane, outputHeight, outputWidth]
    return torch.empty(size, dtype=input.dtype, device=input.device, memory_format=memory_format)
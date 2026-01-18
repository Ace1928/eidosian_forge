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
@register_meta(aten._adaptive_avg_pool2d_backward.default)
def meta__adaptive_avg_pool2d_backward(grad_out, self):
    ndim = grad_out.ndim
    for i in range(1, ndim):
        torch._check(grad_out.size(i) > 0, lambda: f'adaptive_avg_pool2d_backward(): Expected grad_output to have non-zero                       size for non-batch dimensions, {grad_out.shape} with dimension {i} being empty')
    torch._check(ndim == 3 or ndim == 4, lambda: f'adaptive_avg_pool2d_backward(): Expected 3D or 4D tensor, but got {self.shape}')
    torch._check(self.dtype == grad_out.dtype, lambda: f'expected dtype {self.dtype} for `grad_output` but got dtype {grad_out.dtype}')
    memory_format = torch.contiguous_format
    if is_channels_last(self):
        memory_format = torch.channels_last
    return self.new_empty(self.shape).to(memory_format=memory_format)
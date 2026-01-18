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
@register_meta(aten.max_pool2d_with_indices_backward.default)
def meta_max_pool2d_with_indices_backward(grad_output, self, kernel_size, stride, padding, dilation, ceil_mode, indices):
    nInputPlane, outputHeight, outputWidth = max_pool2d_checks_and_compute_shape(self, kernel_size, stride, padding, dilation, ceil_mode)
    torch._check(self.dtype == grad_output.dtype, lambda: f'Expected dtype {self.dtype} for `gradOutput` but got dtype {grad_output.dtype}')
    nOutputPlane = nInputPlane
    ndim = self.ndim

    def _check_dim_size(t):
        check_dim_size(t, ndim, ndim - 3, nOutputPlane)
        check_dim_size(t, ndim, ndim - 2, outputHeight)
        check_dim_size(t, ndim, ndim - 1, outputWidth)
    _check_dim_size(grad_output)
    _check_dim_size(indices)
    memory_format = utils.suggest_memory_format(self)
    return torch.empty(self.shape, dtype=self.dtype, device=self.device, memory_format=memory_format)
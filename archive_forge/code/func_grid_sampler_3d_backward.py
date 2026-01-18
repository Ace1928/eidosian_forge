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
@register_meta(aten.grid_sampler_3d_backward)
@out_wrapper('grad_input', 'grad_grid')
def grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask):
    check_grid_sampler_common(input, grid)
    check_grid_sampler_3d(input, grid, interpolation_mode)
    input_requires_grad = output_mask[0]
    if input_requires_grad:
        grad_input = torch.zeros_like(input, memory_format=torch.legacy_contiguous_format)
    else:
        grad_input = None
    grad_grid = torch.empty_like(grid, memory_format=torch.legacy_contiguous_format)
    return (grad_input, grad_grid)
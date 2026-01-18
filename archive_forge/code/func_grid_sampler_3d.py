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
@register_meta(aten.grid_sampler_3d)
@out_wrapper()
def grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners):
    check_grid_sampler_common(input, grid)
    check_grid_sampler_3d(input, grid, interpolation_mode)
    N = input.shape[0]
    C = input.shape[1]
    out_D = grid.shape[1]
    out_H = grid.shape[2]
    out_W = grid.shape[3]
    return input.new_empty((N, C, out_D, out_H, out_W))
import functools
import logging
import math
import sys
import typing
from typing import Optional
import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import (
from torch._decomp.decompositions import (
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype
from . import config, inductor_prims
@register_decomposition([aten.grid_sampler_2d])
@pw_cast_for_opmath
def grid_sampler_2d(a: torch.Tensor, grid: torch.Tensor, interpolation_mode: int=0, padding_mode: int=0, align_corners: bool=False) -> torch.Tensor:
    _expand_grid = not (a.device == torch.device('cpu') and interpolation_mode == 0 and a.is_contiguous(memory_format=torch.contiguous_format))
    output = decomp_grid_sampler_2d(a, grid=grid, interpolation_mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners, _expand_grid=_expand_grid)
    return output
import functools
import numbers
import operator
import sys
from enum import Enum
from functools import partial, reduce
from itertools import chain, product
from typing import Callable, cast, Iterable, List, Optional, Tuple, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch.nn.functional as F
from torch import sym_float, sym_int, Tensor
from torch._decomp import register_decomposition
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import IntLike, NumberType, TensorLike, TensorSequenceType
from torch._prims_common.wrappers import (
from torch.utils import _pytree as pytree
from torch.utils._pytree import tree_map
@register_decomposition(aten.affine_grid_generator)
@out_wrapper()
@pw_cast_for_opmath
def affine_grid_generator(theta: Tensor, size: List[int], align_corners: bool):
    torch._check(len(size) in (4, 5), lambda: 'affine_grid_generator needs 4d (spatial) or 5d (volumetric) inputs.')
    if len(size) == 4:
        return _affine_grid_generator_4d(theta, size, align_corners=align_corners)
    else:
        return _affine_grid_generator_5d(theta, size, align_corners=align_corners)
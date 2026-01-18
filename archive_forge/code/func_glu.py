import math
from functools import wraps
from typing import Callable, Optional, Union
import torch
import torch._prims as prims
import torch._prims_common as utils
import torch._refs as refs
from torch._decomp import register_decomposition
from torch._prims_common import (
from torch._prims_common.wrappers import (
from torch._refs import _make_inplace
@register_decomposition(aten.glu)
@out_wrapper()
@elementwise_type_promotion_wrapper(type_promoting_args=('a',), type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def glu(a: TensorLikeType, dim: int=-1) -> TensorLikeType:
    dim = utils.canonicalize_dims(a.ndim, dim)
    torch._check(a.shape[dim] % 2 == 0, lambda: f'Halving dimension must be even, but dimension {dim} is size {a.shape[dim]}')
    b, c = torch.tensor_split(a, 2, dim)
    return b * torch.sigmoid(c)
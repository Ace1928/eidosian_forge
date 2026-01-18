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
@register_decomposition(aten.pairwise_distance)
@out_wrapper()
def pairwise_distance(x1: TensorLikeType, x2: TensorLikeType, p: NumberType=2.0, eps: NumberType=1e-06, keepdim=False) -> TensorLikeType:
    return torch.linalg.vector_norm(x1 - x2 + eps, ord=p, dim=-1, keepdim=keepdim)
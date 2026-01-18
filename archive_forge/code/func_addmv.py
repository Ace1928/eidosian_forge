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
@register_decomposition(aten.addmv)
@out_wrapper()
@pw_cast_for_opmath
def addmv(self: Tensor, mat1: Tensor, vec: Tensor, beta: int=1, alpha: int=1):
    if not self.is_floating_point() and (not self.is_complex()):
        beta = int(beta)
        alpha = int(alpha)
    out = alpha * torch.mv(mat1, vec)
    if beta == 0:
        return out
    return out + beta * self
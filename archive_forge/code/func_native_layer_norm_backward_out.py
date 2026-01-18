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
@register_decomposition(aten.native_layer_norm_backward.out)
def native_layer_norm_backward_out(grad_out: Tensor, input: Tensor, normalized_shape: List[int], mean: Tensor, rstd: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], output_mask: List[bool], *, out0: torch.Tensor, out1: torch.Tensor, out2: torch.Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    result = native_layer_norm_backward(grad_out, input, normalized_shape, mean, rstd, weight, bias, output_mask)
    grad_input = (out0, out1, out2)
    for i, r in enumerate(result):
        if r is not None:
            _maybe_resize_out(grad_input[i], r.shape)
            _safe_copy_out(copy_from=r, copy_to=grad_input[i], exact_dtype=True)
    return grad_input
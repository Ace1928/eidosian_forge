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
@register_decomposition(aten.nll_loss2d_backward)
@out_wrapper('grad_input')
def nll_loss2d_backward(grad_output: Tensor, self: Tensor, target: Tensor, weight: Optional[Tensor], reduction: int, ignore_index: int, total_weight: Tensor) -> Tensor:
    assert self.dim() == 4, f'only batches of spatial inputs supported (4D tensors), but got input of dimension: {self.dim()}'
    assert target.dim() == 3, f'only batches of spatial targets supported (3D tensors) but got targets of dimension: {target.dim()}'
    assert self.shape[0] == target.shape[0] and self.shape[2] == target.shape[1] and (self.shape[3] == target.shape[2]), f'size mismatch (got input: {self.shape}, target: {target.shape}'
    assert total_weight.numel() == 1, f'expected total_weight to be a single element tensor, got: {total_weight.shape} ( {total_weight.numel()}, elements)'
    return _nll_loss_backward(grad_output, self, target, weight, reduction, ignore_index, total_weight)
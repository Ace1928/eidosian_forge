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
@register_decomposition(aten.embedding_dense_backward)
@out_wrapper()
def embedding_dense_backward(grad_output: Tensor, indices: Tensor, num_weights: int, padding_idx: int, scale_grad_by_freq: bool):
    computation_dtype, result_dtype = utils.elementwise_dtypes(grad_output, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
    grad_output = grad_output.to(computation_dtype)
    indices = _maybe_convert_to_dtype(indices, torch.long)
    if scale_grad_by_freq:
        counts = indices.new_zeros((num_weights,))
        ones = torch.ones_like(indices)
        counts = aten._unsafe_index_put(counts, [indices], ones, accumulate=True)
        grad_weights_scale = counts[indices]
        grad_output = grad_output / grad_weights_scale.unsqueeze(-1)
    mask = _unsqueeze_to_dim(indices == padding_idx, grad_output.ndim)
    grad = grad_output.masked_fill(mask, 0)
    grad_weight = grad_output.new_zeros((num_weights,) + grad_output.shape[indices.ndim:])
    return aten._unsafe_index_put(grad_weight, [indices], grad, accumulate=True).to(result_dtype)
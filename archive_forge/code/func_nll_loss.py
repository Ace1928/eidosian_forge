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
@register_decomposition(aten.nll_loss)
@out_wrapper()
@elementwise_type_promotion_wrapper(type_promoting_args=('input',), type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
def nll_loss(input: TensorLikeType, target: TensorLikeType, weight: Optional[TensorLikeType]=None, size_average: Optional[bool]=None, ignore_index: int=-100, reduce: Optional[bool]=None, reduction: str='mean') -> TensorLikeType:
    """
    Reference implementation of torch.nn.functional.nll_loss
    """
    torch._check(input.ndim > 0, lambda: f'Expected input tensor to have 1 or more dimensions (got {input.ndim})')
    if size_average is not None or reduce is not None:
        reduction = _get_string_reduction_arg(size_average=size_average, reduce=reduce)
    if input.numel() == 0 and target.numel() == 0:
        if reduction == 'none':
            return torch.zeros_like(target)
        elif reduction == 'sum':
            return torch.empty_like(target)
        else:
            return torch.full_like(target, float('nan'))
    if input.ndim <= 3:
        return _nll_loss_nd(input, target, weight, reduction, ignore_index)
    torch._check(input.ndim > 0 and target.ndim > 0 and (target.shape[1:] == input.shape[2:]), lambda: f'Expected input and target to both have ndim > 0 and target.shape[1:] == input.shape[2:], but got target.shape {target.shape} and input.shape {input.shape}')
    batch_size = input.shape[0]
    num_classes = input.shape[1]
    out_size = [batch_size] + list(target.shape[1:])
    input = torch.reshape(input, [batch_size, num_classes, -1])
    target = torch.reshape(target, [batch_size, -1])
    if reduction != 'none':
        return _nll_loss_nd(input, target, weight, reduction, ignore_index)
    else:
        result = _nll_loss_nd(input, target, weight, reduction, ignore_index)
        return torch.reshape(result, out_size)
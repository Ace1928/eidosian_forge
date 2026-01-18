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
def native_batch_norm_helper(input: Tensor, weight: Optional[Tensor], bias: Optional[Tensor], running_mean: Optional[Tensor], running_var: Optional[Tensor], training: bool, momentum: float, eps: float, functional: bool) -> Tuple[Tensor, Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    reduction_dims = [0] + list(range(2, input.dim()))
    computation_dtype = utils.get_computation_dtype(input.dtype)
    new_running_mean = running_mean
    new_running_var = running_var
    if training:
        computation_dtype = utils.get_computation_dtype(input.dtype)
        input_acc = input.to(dtype=computation_dtype)
        biased_var, mean = torch.var_mean(input_acc, dim=reduction_dims, correction=0, keepdim=True)
        rstd = torch.rsqrt(biased_var + eps)
        output = (input - mean) * rstd
        save_mean = torch.squeeze(mean, reduction_dims)
        save_rstd = torch.squeeze(rstd, reduction_dims)
        if running_mean is not None:
            new_running_mean = momentum * save_mean + (1 - momentum) * running_mean
            if not functional:
                running_mean.copy_(new_running_mean)
        if running_var is not None:
            n = input.numel() / input.shape[1]
            squeezed_var = torch.squeeze(biased_var, reduction_dims)
            unbiased_var = squeezed_var * (n / (n - 1))
            new_running_var = momentum * unbiased_var + (1 - momentum) * running_var
            if not functional:
                running_var.copy_(new_running_var)
    else:
        assert running_mean is not None and running_var is not None
        running_mean = running_mean.to(dtype=computation_dtype, copy=True)
        new_running_mean = running_mean
        running_var = running_var.to(dtype=computation_dtype, copy=True)
        new_running_var = running_var
        mean = running_mean
        invstd = 1 / torch.sqrt(running_var + eps)
        if input.device.type != 'cpu':
            save_mean = running_mean
            save_rstd = invstd
        else:
            save_mean = input.new_zeros((0,))
            save_rstd = input.new_zeros((0,))
        mean = _unsqueeze_to_dim(mean, input.dim() - 1)
        invstd = _unsqueeze_to_dim(invstd, input.dim() - 1)
        output = (input - mean) * invstd
    if weight is not None:
        weight = weight.flatten()
        weight = _unsqueeze_to_dim(weight, input.dim() - 1)
        output = output * weight
    if bias is not None:
        bias = bias.flatten()
        bias = _unsqueeze_to_dim(bias, input.dim() - 1)
        output = output + bias
    if input.device.type == 'cpu':
        save_mean = save_mean.to(dtype=input.dtype)
        save_rstd = save_rstd.to(dtype=input.dtype)
    return (output.to(dtype=input.dtype), save_mean, save_rstd, new_running_mean, new_running_var)
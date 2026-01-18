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
@register_decomposition(aten.native_group_norm_backward.default)
@pw_cast_for_opmath
def native_group_norm_backward(grad_output: Tensor, input: Tensor, mean: Tensor, rstd: Tensor, gamma: Optional[Tensor], N: int, C: int, HxW: int, group: int, output_mask: List[bool]) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
    utils.check_same_device(grad_output, input, mean, rstd, allow_cpu_scalar_tensors=False)
    utils.check_same_shape(input, grad_output, allow_cpu_scalar_tensors=False)
    utils.check_same_shape(mean, rstd, allow_cpu_scalar_tensors=False)
    torch._check(input.numel() == N * C * HxW, lambda: f'Expect input to have {N * C * HxW} elements')
    torch._check(mean.shape == (N, group), lambda: f'Expect mean to have shape ({N}, {group}, but got {mean.shape}')
    torch._check(gamma is None or gamma.numel() == C, lambda: f'Expect gamma to have {C} elements but got {(gamma.numel() if gamma is not None else -1)}')
    cpg, _rem = divmod(C, group)
    torch._check(_rem == 0, lambda: f'Expect number of channels {C} to be evenly-divisible by number of groups {group}')
    ds = torch.mul(grad_output, input).view(N, C, HxW).sum(dim=[2])
    db = grad_output.view(N, C, HxW).sum(dim=[2])
    d_input: Optional[Tensor] = None
    d_gamma: Optional[Tensor] = None
    d_bias: Optional[Tensor] = None
    if output_mask[0]:
        s = 1.0 / (HxW * cpg)
        if gamma is not None:
            ds_val = torch.mul(ds, gamma.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            db_val = torch.mul(db, gamma.unsqueeze(0)).reshape(N, group, cpg).sum(2)
            c1 = torch.mul(rstd.unsqueeze(-1), gamma.reshape(1, group, cpg))
        else:
            ds_val = ds.reshape(N, group, cpg).sum(2)
            db_val = db.reshape(N, group, cpg).sum(2)
            c1 = torch.mul(rstd.unsqueeze(-1), torch.ones((1, group, cpg), device=rstd.device))
        c2 = (db_val * mean - ds_val) * rstd * rstd * rstd * s
        c3 = -c2 * mean - db_val * rstd * s
        c1 = c1.unsqueeze(-1)
        c2 = _unsqueeze_to_dim(c2, 4)
        c3 = _unsqueeze_to_dim(c3, 4)
        d_input = torch.mul(grad_output.reshape(N, group, cpg, HxW), c1) + torch.mul(input.reshape(N, group, cpg, HxW), c2) + c3
        d_input = d_input.reshape(input.shape).to(input.dtype)
    if output_mask[1]:
        d_gamma = ((ds.view(N, group, cpg) - db.view(N, group, cpg) * mean.unsqueeze(-1)) * rstd.unsqueeze(-1)).sum(dim=[0]).reshape(C)
    if output_mask[2]:
        d_bias = db.sum(dim=[0])
    return (d_input, d_gamma, d_bias)
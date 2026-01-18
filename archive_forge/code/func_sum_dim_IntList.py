import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.sum.dim_IntList, 'self: jt, dim: any?, keepdim: any?, dtype: any?')
def sum_dim_IntList(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    assert inp._ragged_idx == 1
    new_kwargs['dim'], ragged_reduced_away = _wrap_jagged_dims(inp.dim(), new_kwargs['dim'], 'sum')
    if not ragged_reduced_away:
        return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))
    else:
        out = func(inp._values, **new_kwargs)
        if new_kwargs['keepdim']:
            out = out.unsqueeze(0)
        return out
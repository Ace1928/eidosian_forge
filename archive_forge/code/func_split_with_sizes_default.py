import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.split_with_sizes.default, 'self: jt, split_sizes: any, dim: any')
def split_with_sizes_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    new_kwargs['dim'] = _wrap_jagged_dim(inp.dim(), new_kwargs['dim'], 'split_with_sizes')
    return [NestedTensor(values=x, **extract_kwargs(inp)) for x in func(inp._values, **new_kwargs)]
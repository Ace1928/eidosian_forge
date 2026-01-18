import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.linear.default, 'input: jt, weight: t, bias: t?')
def linear_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    weight = new_kwargs['weight']
    bias = new_kwargs['bias']
    return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))
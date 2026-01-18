import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.native_dropout.default, 'self: jt, float: any, train: any?')
def native_dropout_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    out1, out2 = func(inp._values, **new_kwargs)
    return (NestedTensor(out1, **extract_kwargs(inp)), NestedTensor(out2, **extract_kwargs(inp)))
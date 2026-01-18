import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.native_layer_norm.default, 'input: jt, normalized_shape: any, weight: any?, bias: any?, eps: any')
def native_layer_norm_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    normalized_shape = new_kwargs['normalized_shape']
    if inp.dim() < 3 or inp.dim() - len(normalized_shape) < 2:
        raise RuntimeError('layer_norm(): normalizing over ragged dim not supported for nested tensors')
    output, mean, std = func(inp._values, **new_kwargs)
    return (NestedTensor(output, **extract_kwargs(inp)), mean, std)
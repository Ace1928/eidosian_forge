import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.stack.default, 'tensors: any, dim: any')
def stack_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    tensors = new_kwargs.pop('tensors')
    for t in tensors:
        if not isinstance(t, NestedTensor):
            raise RuntimeError('stack(): expected all nested tensors inputs')
        if t.dim() != tensors[0].dim():
            raise RuntimeError('stack(): expected all nested tensors to have the same dim')
        if not raggedness_matches(t, tensors[0].shape):
            raise RuntimeError('stack(): expected all nested tensors to have the same nested structure')
    new_kwargs['dim'] = _wrap_jagged_dim(tensors[0].dim() + 1, new_kwargs['dim'], 'stack')
    return NestedTensor(func([t._values for t in tensors], **new_kwargs), **extract_kwargs(tensors[0]))
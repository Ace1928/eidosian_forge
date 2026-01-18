import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def jagged_torch_function(func, *args, **kwargs):
    if func is torch._C._nn.scaled_dot_product_attention:
        return jagged_scaled_dot_product_attention(*args, **kwargs)
    if func.__name__ == 'flatten':

        def _flatten_sig(input, start_dim=0, end_dim=-1):
            pass
        _, new_kwargs = normalize_function(_flatten_sig, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
        inp = new_kwargs.pop('input')
        new_kwargs['start_dim'] = _wrap_jagged_dim(inp.dim(), new_kwargs['start_dim'], 'flatten')
        new_kwargs['end_dim'] = _wrap_jagged_dim(inp.dim(), new_kwargs['end_dim'], 'flatten')
        return NestedTensor(func(inp._values, **new_kwargs), **extract_kwargs(inp))
    raise NotImplementedError(func)
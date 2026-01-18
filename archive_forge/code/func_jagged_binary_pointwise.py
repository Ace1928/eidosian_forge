import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def jagged_binary_pointwise(func, *args, **kwargs):
    a, b = (args[0], args[1])
    assert isinstance(a, NestedTensor) or isinstance(b, NestedTensor)
    mismatch_error_msg = f'cannot call binary pointwise function {func.__name__} with inputs of shapes {a.shape} and {b.shape}'
    if isinstance(a, NestedTensor) and isinstance(b, NestedTensor):
        if raggedness_matches(a, b.shape):
            return NestedTensor(func(a._values, b._values, *args[2:], **kwargs), **extract_kwargs(a))
        raise RuntimeError(mismatch_error_msg)
    a_is_nt = isinstance(a, NestedTensor)
    extracted_kwargs = extract_kwargs(a) if a_is_nt else extract_kwargs(b)
    nt, t = (a, b) if a_is_nt else (b, a)
    if t.dim() > nt.dim():
        raise NotImplementedError('NYI: broadcasting NT with T with larger dim')
    t_squeezed = squeeze_leading_ones(t)
    if nt.dim() >= t_squeezed.dim() + 2:
        lhs, rhs = (nt._values, t_squeezed) if a_is_nt else (t_squeezed, nt._values)
        return NestedTensor(func(lhs, rhs, *args[2:], **kwargs), **extracted_kwargs)
    if a.dim() == b.dim():
        if a.shape[0] != b.shape[0]:
            raise RuntimeError(mismatch_error_msg)
        outputs = []
        for a_comp, b_comp in zip(a.unbind(), b.unbind()):
            outputs.append(func(a_comp, b_comp, *args[2:], **kwargs))
        new_values = torch.cat(outputs, dim=0)
        return NestedTensor(new_values, **extracted_kwargs)
    raise RuntimeError(mismatch_error_msg)
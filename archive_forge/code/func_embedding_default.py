import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.embedding.default, 'weight: t, indices: jt, padding_idx: any?, scale_grad_by_freq: any?, sparse: any?')
def embedding_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    indices = new_kwargs.pop('indices')
    weight = new_kwargs.pop('weight')
    return NestedTensor(func(weight, indices._values, **new_kwargs), **extract_kwargs(indices))
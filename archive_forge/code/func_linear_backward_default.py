import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
@register_jagged_func(torch.ops.aten.linear_backward.default, 'self: jt, grad_output: jt, weight: t, output_mask: any')
def linear_backward_default(func, *args, **kwargs):
    _, new_kwargs = normalize_function(func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True)
    inp = new_kwargs.pop('input')
    grad_output = new_kwargs.pop('grad_output')
    weight = new_kwargs.pop('weight')
    check_ragged_dim_same(func, inp, 'self', grad_output, 'grad_output')
    ds = NestedTensor(torch.mm(grad_output._values, weight), **extract_kwargs(grad_output))
    dw = torch.mm(grad_output._values.T, inp._values)
    db = None
    return (ds, dw, db)
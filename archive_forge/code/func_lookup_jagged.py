import functools
import math
import torch
from torch.nested._internal.sdpa import jagged_scaled_dot_product_attention
from .nested_tensor import NestedTensor
from typing import *  # noqa: F403
from torch.fx.operator_schemas import normalize_function
def lookup_jagged(func, *args, **kwargs) -> Optional[Callable]:
    dispatch_func = JAGGED_OPS_TABLE.get(func, None)
    if dispatch_func is not None:
        return dispatch_func
    if torch.Tag.pointwise in func.tags:
        num_tensor_args = sum([isinstance(x, torch.Tensor) for x in args])
        if num_tensor_args == 1:
            return functools.partial(jagged_unary_pointwise, func)
        elif num_tensor_args == 2:
            check_schema('lhs: any, rhs: any', func, *args, **kwargs)
            return functools.partial(jagged_binary_pointwise, func)
    return None
import operator
import torch
from . import _dtypes_impl
def apply_keepdims(tensor, axis, ndim):
    if axis is None:
        shape = (1,) * ndim
        tensor = tensor.expand(shape).contiguous()
    else:
        shape = expand_shape(tensor.shape, axis)
        tensor = tensor.reshape(shape)
    return tensor
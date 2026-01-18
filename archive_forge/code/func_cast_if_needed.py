import operator
import torch
from . import _dtypes_impl
def cast_if_needed(tensor, dtype):
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    return tensor
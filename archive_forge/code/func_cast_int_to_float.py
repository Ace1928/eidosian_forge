import operator
import torch
from . import _dtypes_impl
def cast_int_to_float(x):
    if _dtypes_impl._category(x.dtype) < 2:
        x = x.to(_dtypes_impl.default_dtypes().float_dtype)
    return x
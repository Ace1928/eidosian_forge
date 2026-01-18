from __future__ import annotations
import operator
import numpy as np
from xarray.core import dtypes, duck_array_ops
def inject_numpy_same(cls):
    for name in NUMPY_SAME_METHODS:
        setattr(cls, name, _values_method_wrapper(name))
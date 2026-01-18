import functools
import warnings
import numpy as np
from numba import jit, typeof
from numba.core import cgutils, types, serialize, sigutils, errors
from numba.core.extending import (is_jitted, overload_attribute,
from numba.core.typing import npydecl
from numba.core.typing.templates import AbstractTemplate, signature
from numba.cpython.unsafe.tuple import tuple_setitem
from numba.np.ufunc import _internal
from numba.parfors import array_analysis
from numba.np.ufunc import ufuncbuilder
from numba.np import numpy_support
from typing import Callable
from llvmlite import ir
def impl_nd_axis_tuple(ufunc, array, axis=0, dtype=None, initial=None):
    axis_ = fixup_axis(axis, array.ndim)
    for i in range(0, len(axis_)):
        if axis_[i] < 0 or axis_[i] >= array.ndim:
            raise ValueError('Invalid axis')
        for j in range(i + 1, len(axis_)):
            if axis_[i] == axis_[j]:
                raise ValueError("duplicate value in 'axis'")
    min_idx, min_elem = find_min(axis_)
    r = ufunc.reduce(array, axis=min_elem, dtype=dtype, initial=initial)
    if len(axis) == 1:
        return r
    elif len(axis) == 2:
        return ufunc.reduce(r, axis=axis_[(min_idx + 1) % 2] - 1)
    else:
        ax = axis_tup
        for i in range(len(ax)):
            if i != min_idx:
                ax = tuple_setitem(ax, i, axis_[i])
        return ufunc.reduce(r, axis=ax)
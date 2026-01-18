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
@register_jitable
def fixup_axis(axis, ndim):
    ax = axis
    for i in range(len(axis)):
        val = axis[i] + ndim if axis[i] < 0 else axis[i]
        ax = tuple_setitem(ax, i, val)
    return ax
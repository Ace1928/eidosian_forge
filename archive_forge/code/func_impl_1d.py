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
def impl_1d(ufunc, array, axis=0, dtype=None, initial=None):
    start = 0
    if init_none and id_none:
        start = 1
        r = array[0]
    elif init_none:
        r = identity
    else:
        r = initial
    sz = array.shape[0]
    for i in range(start, sz):
        r = ufunc(r, array[i])
    return r
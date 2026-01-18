import math
from collections import namedtuple
import operator
import warnings
import llvmlite.ir
import numpy as np
from numba.core import types, cgutils
from numba.core.extending import overload, overload_method, register_jitable
from numba.np.numpy_support import (as_dtype, type_can_asarray, type_is_scalar,
from numba.core.imputils import (lower_builtin, impl_ret_borrowed,
from numba.np.arrayobj import (make_array, load_item, store_item,
from numba.np.linalg import ensure_blas
from numba.core.extending import intrinsic
from numba.core.errors import (RequireLiteralValue, TypingError,
from numba.cpython.unsafe.tuple import tuple_setitem
@register_jitable
def less_than_or_equal_complex(a, b):
    if np.isnan(a.real):
        if np.isnan(b.real):
            if np.isnan(a.imag):
                return np.isnan(b.imag)
            elif np.isnan(b.imag):
                return True
            else:
                return a.imag <= b.imag
        else:
            return False
    elif np.isnan(b.real):
        return True
    elif np.isnan(a.imag):
        if np.isnan(b.imag):
            return a.real <= b.real
        else:
            return False
    elif np.isnan(b.imag):
        return True
    else:
        if a.real < b.real:
            return True
        elif a.real == b.real:
            return a.imag <= b.imag
        return False
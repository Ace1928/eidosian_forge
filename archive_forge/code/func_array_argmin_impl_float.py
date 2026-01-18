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
def array_argmin_impl_float(arry):
    if arry.size == 0:
        raise ValueError('attempt to get argmin of an empty sequence')
    for v in arry.flat:
        min_value = v
        min_idx = 0
        break
    if np.isnan(min_value):
        return min_idx
    idx = 0
    for v in arry.flat:
        if np.isnan(v):
            return idx
        if v < min_value:
            min_value = v
            min_idx = idx
        idx += 1
    return min_idx
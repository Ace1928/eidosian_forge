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
def array_argmax_impl_datetime(arry):
    if arry.size == 0:
        raise ValueError('attempt to get argmax of an empty sequence')
    it = np.nditer(arry)
    max_value = next(it).take(0)
    max_idx = 0
    if np.isnat(max_value):
        return max_idx
    idx = 1
    for view in it:
        v = view.item()
        if np.isnat(v):
            return idx
        if v > max_value:
            max_value = v
            max_idx = idx
        idx += 1
    return max_idx
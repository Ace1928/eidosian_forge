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
def np_extract_impl(condition, arr):
    cond = np.asarray(condition).flatten()
    a = np.asarray(arr)
    if a.size == 0:
        raise ValueError('Cannot extract from an empty array')
    if np.any(cond[a.size:]) and cond.size > a.size:
        msg = 'condition shape inconsistent with arr shape'
        raise ValueError(msg)
    max_len = min(a.size, cond.size)
    out = [a.flat[idx] for idx in range(max_len) if cond[idx]]
    return np.array(out)
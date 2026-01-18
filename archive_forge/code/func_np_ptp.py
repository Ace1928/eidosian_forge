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
@overload_method(types.Array, 'ptp')
@overload(np.ptp)
def np_ptp(a):
    if hasattr(a, 'dtype'):
        if isinstance(a.dtype, types.Boolean):
            raise TypingError('Boolean dtype is unsupported (as per NumPy)')

    def np_ptp_impl(a):
        arr = prepare_ptp_input(a)
        a_flat = arr.flat
        a_min = a_flat[0]
        a_max = a_flat[0]
        for i in range(arr.size):
            val = a_flat[i]
            take_branch, retval = _early_return(val)
            if take_branch:
                return retval
            a_max = _compute_a_max(a_max, val)
            a_min = _compute_a_min(a_min, val)
        return a_max - a_min
    return np_ptp_impl
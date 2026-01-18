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
@overload(np.interp)
def np_interp(x, xp, fp):
    if hasattr(xp, 'ndim') and xp.ndim > 1:
        raise TypingError('xp must be 1D')
    if hasattr(fp, 'ndim') and fp.ndim > 1:
        raise TypingError('fp must be 1D')
    complex_dtype_msg = 'Cannot cast array data from complex dtype to float64 dtype'
    xp_dt = determine_dtype(xp)
    if np.issubdtype(xp_dt, np.complexfloating):
        raise TypingError(complex_dtype_msg)
    fp_dt = determine_dtype(fp)
    dtype = np.result_type(fp_dt, np.float64)
    if np.issubdtype(dtype, np.complexfloating):
        inner = np_interp_impl_complex_inner
    else:
        inner = np_interp_impl_inner

    def np_interp_impl(x, xp, fp):
        return inner(x, xp, fp, dtype)

    def np_interp_scalar_impl(x, xp, fp):
        return inner(x, xp, fp, dtype).flat[0]
    if isinstance(x, types.Number):
        if isinstance(x, types.Complex):
            raise TypingError(complex_dtype_msg)
        return np_interp_scalar_impl
    return np_interp_impl
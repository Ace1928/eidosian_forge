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
@overload(np.isclose)
def isclose(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
    if not type_can_asarray(a):
        raise TypingError('The first argument "a" must be array-like')
    if not type_can_asarray(b):
        raise TypingError('The second argument "b" must be array-like')
    if not isinstance(rtol, (float, types.Float)):
        raise TypingError('The third argument "rtol" must be a floating point')
    if not isinstance(atol, (float, types.Float)):
        raise TypingError('The fourth argument "atol" must be a floating point')
    if not isinstance(equal_nan, (bool, types.Boolean)):
        raise TypingError('The fifth argument "equal_nan" must be a boolean')
    if isinstance(a, types.Array) and isinstance(b, types.Number):

        def isclose_impl(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
            x = a.reshape(-1)
            y = b
            out = np.zeros(len(x), np.bool_)
            for i in range(len(out)):
                out[i] = _isclose_item(x[i], y, rtol, atol, equal_nan)
            return out.reshape(a.shape)
    elif isinstance(a, types.Number) and isinstance(b, types.Array):

        def isclose_impl(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
            x = a
            y = b.reshape(-1)
            out = np.zeros(len(y), np.bool_)
            for i in range(len(out)):
                out[i] = _isclose_item(x, y[i], rtol, atol, equal_nan)
            return out.reshape(b.shape)
    elif isinstance(a, types.Array) and isinstance(b, types.Array):

        def isclose_impl(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
            shape = np.broadcast_shapes(a.shape, b.shape)
            a_ = np.broadcast_to(a, shape)
            b_ = np.broadcast_to(b, shape)
            out = np.zeros(len(a_), dtype=np.bool_)
            for i, (av, bv) in enumerate(np.nditer((a_, b_))):
                out[i] = _isclose_item(av.item(), bv.item(), rtol, atol, equal_nan)
            return np.broadcast_to(out, shape)
    else:

        def isclose_impl(a, b, rtol=1e-05, atol=1e-08, equal_nan=False):
            return _isclose_item(a, b, rtol, atol, equal_nan)
    return isclose_impl
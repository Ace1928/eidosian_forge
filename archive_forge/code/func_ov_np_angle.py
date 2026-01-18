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
@overload(np.angle)
def ov_np_angle(z, deg=False):
    deg_mult = float(180 / np.pi)
    if isinstance(z, types.Number):

        def impl(z, deg=False):
            if deg:
                return np.arctan2(z.imag, z.real) * deg_mult
            else:
                return np.arctan2(z.imag, z.real)
        return impl
    elif isinstance(z, types.Array):
        dtype = z.dtype
        if isinstance(dtype, types.Complex):
            ret_dtype = dtype.underlying_float
        elif isinstance(dtype, types.Float):
            ret_dtype = dtype
        else:
            return

        def impl(z, deg=False):
            out = np.zeros_like(z, dtype=ret_dtype)
            for index, val in np.ndenumerate(z):
                out[index] = np.angle(val, deg)
            return out
        return impl
    else:
        raise NumbaTypeError(f'Argument "z" must be a complex or Array[complex]. Got {z}')
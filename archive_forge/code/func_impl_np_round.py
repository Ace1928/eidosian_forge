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
@overload(np.around)
@overload(np.round)
@overload(np.round_)
def impl_np_round(a, decimals=0, out=None):
    if not type_can_asarray(a):
        raise TypingError('The argument "a" must be array-like')
    if not (isinstance(out, types.Array) or is_nonelike(out)):
        msg = 'The argument "out" must be an array if it is provided'
        raise TypingError(msg)
    if isinstance(a, (types.Float, types.Integer, types.Complex)):
        if is_nonelike(out):
            if isinstance(a, types.Float):

                def impl(a, decimals=0, out=None):
                    if decimals == 0:
                        return _np_round_float(a)
                    else:
                        return round_ndigits(a, decimals)
                return impl
            elif isinstance(a, types.Integer):

                def impl(a, decimals=0, out=None):
                    if decimals == 0:
                        return a
                    else:
                        return int(round_ndigits(a, decimals))
                return impl
            elif isinstance(a, types.Complex):

                def impl(a, decimals=0, out=None):
                    if decimals == 0:
                        real = _np_round_float(a.real)
                        imag = _np_round_float(a.imag)
                    else:
                        real = round_ndigits(a.real, decimals)
                        imag = round_ndigits(a.imag, decimals)
                    return complex(real, imag)
                return impl
        else:

            def impl(a, decimals=0, out=None):
                out[0] = np.round(a, decimals)
                return out
            return impl
    elif isinstance(a, types.Array):
        if is_nonelike(out):

            def impl(a, decimals=0, out=None):
                out = np.empty_like(a)
                return np.round(a, decimals, out)
            return impl
        else:

            def impl(a, decimals=0, out=None):
                if a.shape != out.shape:
                    raise ValueError('invalid output shape')
                for index, val in np.ndenumerate(a):
                    out[index] = np.round(val, decimals)
                return out
            return impl
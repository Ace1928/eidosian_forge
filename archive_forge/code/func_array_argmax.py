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
@overload(np.argmax)
@overload_method(types.Array, 'argmax')
def array_argmax(a, axis=None):
    if isinstance(a.dtype, (types.NPDatetime, types.NPTimedelta)):
        flatten_impl = array_argmax_impl_datetime
    elif isinstance(a.dtype, types.Float):
        flatten_impl = array_argmax_impl_float
    else:
        flatten_impl = array_argmax_impl_generic
    if is_nonelike(axis):

        def array_argmax_impl(a, axis=None):
            return flatten_impl(a)
    else:
        array_argmax_impl = build_argmax_or_argmin_with_axis_impl(a, axis, flatten_impl)
    return array_argmax_impl
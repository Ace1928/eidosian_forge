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
@overload(np.asarray)
def np_asarray(a, dtype=None):
    if not type_can_asarray(a):
        return None
    if isinstance(a, types.Array):
        if is_nonelike(dtype) or a.dtype == dtype.dtype:

            def impl(a, dtype=None):
                return a
        else:

            def impl(a, dtype=None):
                return a.astype(dtype)
    elif isinstance(a, (types.Sequence, types.Tuple)):
        if is_nonelike(dtype):

            def impl(a, dtype=None):
                return np.array(a)
        else:

            def impl(a, dtype=None):
                return np.array(a, dtype)
    elif isinstance(a, (types.Number, types.Boolean)):
        dt_conv = a if is_nonelike(dtype) else dtype
        ty = as_dtype(dt_conv)

        def impl(a, dtype=None):
            return np.array(a, ty)
    elif isinstance(a, types.containers.ListType):
        if not isinstance(a.dtype, (types.Number, types.Boolean)):
            raise TypingError('asarray support for List is limited to Boolean and Number types')
        target_dtype = a.dtype if is_nonelike(dtype) else dtype

        def impl(a, dtype=None):
            l = len(a)
            ret = np.empty(l, dtype=target_dtype)
            for i, v in enumerate(a):
                ret[i] = v
            return ret
    elif isinstance(a, types.StringLiteral):
        arr = np.asarray(a.literal_value)

        def impl(a, dtype=None):
            return arr.copy()
    else:
        impl = None
    return impl
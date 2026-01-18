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
@overload(np.argwhere)
def np_argwhere(a):
    use_scalar = isinstance(a, (types.Number, types.Boolean))
    if type_can_asarray(a) and (not use_scalar):

        def impl(a):
            arr = np.asarray(a)
            if arr.shape == ():
                return np.zeros((0, 1), dtype=types.intp)
            return np.transpose(np.vstack(np.nonzero(arr)))
    else:
        falseish = (0, 0)
        trueish = (1, 0)

        def impl(a):
            if a is not None and bool(a):
                return np.zeros(trueish, dtype=types.intp)
            else:
                return np.zeros(falseish, dtype=types.intp)
    return impl
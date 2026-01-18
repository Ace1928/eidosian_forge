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
@overload(np.flatnonzero)
def np_flatnonzero(a):
    if type_can_asarray(a):

        def impl(a):
            arr = np.asarray(a)
            return np.nonzero(np.ravel(arr))[0]
    else:

        def impl(a):
            if a is not None and bool(a):
                data = [0]
            else:
                data = [x for x in range(0)]
            return np.array(data, dtype=types.intp)
    return impl
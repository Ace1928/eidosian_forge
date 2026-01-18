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
def is_np_inf_impl(x, out, fn):
    if is_nonelike(out):

        def impl(x, out=None):
            return np.logical_and(np.isinf(x), fn(np.signbit(x)))
    else:

        def impl(x, out=None):
            return np.logical_and(np.isinf(x), fn(np.signbit(x)), out)
    return impl
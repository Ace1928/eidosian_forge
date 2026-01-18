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
def generate_xinfo_body(arg, np_func, container, attr):
    nbty = getattr(arg, 'dtype', arg)
    np_dtype = as_dtype(nbty)
    try:
        f = np_func(np_dtype)
    except ValueError:
        return None
    data = tuple([getattr(f, x) for x in attr])

    @register_jitable
    def impl(arg):
        return container(*data)
    return impl
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
def nan_min_max_factory(comparison_op, is_complex_dtype):
    if is_complex_dtype:

        def impl(a):
            arr = np.asarray(a)
            check_array(arr)
            it = np.nditer(arr)
            return_val = next(it).take(0)
            for view in it:
                v = view.item()
                if np.isnan(return_val.real) and (not np.isnan(v.real)):
                    return_val = v
                elif comparison_op(v.real, return_val.real):
                    return_val = v
                elif v.real == return_val.real:
                    if comparison_op(v.imag, return_val.imag):
                        return_val = v
            return return_val
    else:

        def impl(a):
            arr = np.asarray(a)
            check_array(arr)
            it = np.nditer(arr)
            return_val = next(it).take(0)
            for view in it:
                v = view.item()
                if not np.isnan(v):
                    if not comparison_op(return_val, v):
                        return_val = v
            return return_val
    return impl
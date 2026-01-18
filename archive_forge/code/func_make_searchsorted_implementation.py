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
def make_searchsorted_implementation(np_dtype, side):
    assert side in VALID_SEARCHSORTED_SIDES
    lt = _less_than
    le = _less_than_or_equal
    if side == 'left':
        _impl = _searchsorted(lt, lt)
    elif np.issubdtype(np_dtype, np.inexact) and numpy_version < (1, 23):
        _impl = _searchsorted(lt, le)
    else:
        _impl = _searchsorted(le, le)
    return register_jitable(_impl)
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
@register_jitable
def round_ndigits(x, ndigits):
    if math.isinf(x) or math.isnan(x):
        return x
    if ndigits >= 0:
        if ndigits > 22:
            pow1 = 10.0 ** (ndigits - 22)
            pow2 = 1e+22
        else:
            pow1 = 10.0 ** ndigits
            pow2 = 1.0
        y = x * pow1 * pow2
        if math.isinf(y):
            return x
        return _np_round_float(y) / pow2 / pow1
    else:
        pow1 = 10.0 ** (-ndigits)
        y = x / pow1
        return _np_round_float(y) * pow1
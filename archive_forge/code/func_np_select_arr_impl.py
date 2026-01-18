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
def np_select_arr_impl(condlist, choicelist, default=0):
    if len(condlist) != len(choicelist):
        raise ValueError('list of cases must be same length as list of conditions')
    out = default * np.ones(choicelist[0].shape, choicelist[0].dtype)
    for i in range(len(condlist) - 1, -1, -1):
        cond = condlist[i]
        choice = choicelist[i]
        out = np.where(cond, choice, out)
    return out
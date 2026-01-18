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
@overload(np.select)
def np_select(condlist, choicelist, default=0):

    def np_select_arr_impl(condlist, choicelist, default=0):
        if len(condlist) != len(choicelist):
            raise ValueError('list of cases must be same length as list of conditions')
        out = default * np.ones(choicelist[0].shape, choicelist[0].dtype)
        for i in range(len(condlist) - 1, -1, -1):
            cond = condlist[i]
            choice = choicelist[i]
            out = np.where(cond, choice, out)
        return out
    if not isinstance(condlist, (types.List, types.UniTuple)):
        raise NumbaTypeError('condlist must be a List or a Tuple')
    if not isinstance(choicelist, (types.List, types.UniTuple)):
        raise NumbaTypeError('choicelist must be a List or a Tuple')
    if not isinstance(default, (int, types.Number, types.Boolean)):
        raise NumbaTypeError('default must be a scalar (number or boolean)')
    if not isinstance(condlist[0], types.Array):
        raise NumbaTypeError('items of condlist must be arrays')
    if not isinstance(choicelist[0], types.Array):
        raise NumbaTypeError('items of choicelist must be arrays')
    if isinstance(condlist[0], types.Array):
        if not isinstance(condlist[0].dtype, types.Boolean):
            raise NumbaTypeError('condlist arrays must contain booleans')
    if isinstance(condlist[0], types.UniTuple):
        if not (isinstance(condlist[0], types.UniTuple) and isinstance(condlist[0][0], types.Boolean)):
            raise NumbaTypeError('condlist tuples must only contain booleans')
    if isinstance(condlist[0], types.Array) and condlist[0].ndim != choicelist[0].ndim:
        raise NumbaTypeError('condlist and choicelist elements must have the same number of dimensions')
    if isinstance(condlist[0], types.Array) and condlist[0].ndim < 1:
        raise NumbaTypeError('condlist arrays must be of at least dimension 1')
    return np_select_arr_impl
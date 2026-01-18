import functools
import math
import operator
from llvmlite import ir
from llvmlite.ir import Constant
import numpy as np
from numba import pndindex, literal_unroll
from numba.core import types, typing, errors, cgutils, extending
from numba.np.numpy_support import (as_dtype, from_dtype, carray, farray,
from numba.np.numpy_support import type_can_asarray, is_nonelike, numpy_version
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core.typing import signature
from numba.core.types import StringLiteral
from numba.core.extending import (register_jitable, overload, overload_method,
from numba.misc import quicksort, mergesort
from numba.cpython import slicing
from numba.cpython.unsafe.tuple import tuple_setitem, build_full_slice_tuple
from numba.core.extending import overload_classmethod
from numba.core.typing.npydecl import (parse_dtype as ty_parse_dtype,
@overload(np.empty_like)
def ol_np_empty_like(arr, dtype=None):
    _check_const_str_dtype('empty_like', dtype)
    if not is_nonelike(dtype):
        nb_dtype = ty_parse_dtype(dtype)
    elif isinstance(arr, types.Array):
        nb_dtype = arr.dtype
    else:
        nb_dtype = arr
    if nb_dtype is not None:
        if isinstance(arr, types.Array):
            layout = arr.layout if arr.layout != 'A' else 'C'
            retty = arr.copy(dtype=nb_dtype, layout=layout, readonly=False)
        else:
            retty = types.Array(nb_dtype, 0, 'C')
    else:
        msg = f'Cannot parse input types to function np.empty_like({arr}, {dtype})'
        raise errors.TypingError(msg)

    def impl(arr, dtype=None):
        return numpy_empty_like_nd(arr, dtype, retty)
    return impl
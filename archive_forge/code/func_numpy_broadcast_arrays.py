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
@overload(np.broadcast_arrays)
def numpy_broadcast_arrays(*args):
    for idx, arg in enumerate(args):
        if not type_can_asarray(arg):
            raise errors.TypingError(f'Argument "{idx}" must be array-like')
    unified_dtype = None
    dt = None
    for arg in args:
        if isinstance(arg, (types.Array, types.BaseTuple)):
            dt = arg.dtype
        else:
            dt = arg
        if unified_dtype is None:
            unified_dtype = dt
        elif unified_dtype != dt:
            raise errors.TypingError(f'Mismatch of argument types. Numba cannot broadcast arrays with different types. Got {args}')
    m = 0
    for idx, arg in enumerate(args):
        if isinstance(arg, types.ArrayCompatible):
            m = max(m, arg.ndim)
        elif isinstance(arg, (types.Number, types.Boolean, types.BaseTuple)):
            m = max(m, 1)
        else:
            raise errors.TypingError(f'Unhandled type {arg}')
    tup_init = (0,) * m

    def impl(*args):
        shape = [1] * m
        for array in literal_unroll(args):
            numpy_broadcast_shapes_list(shape, m, np.asarray(array).shape)
        tup = tup_init
        for i in range(m):
            tup = tuple_setitem(tup, i, shape[i])
        outs = []
        for array in literal_unroll(args):
            outs.append(np.broadcast_to(np.asarray(array), tup))
        return outs
    return impl
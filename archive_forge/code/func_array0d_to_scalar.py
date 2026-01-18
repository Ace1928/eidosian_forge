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
@lower_cast(types.Array, types.UnicodeCharSeq)
@lower_cast(types.Array, types.Float)
@lower_cast(types.Array, types.Integer)
@lower_cast(types.Array, types.Complex)
@lower_cast(types.Array, types.Boolean)
@lower_cast(types.Array, types.NPTimedelta)
@lower_cast(types.Array, types.NPDatetime)
def array0d_to_scalar(context, builder, fromty, toty, val):

    def impl(a):
        return a.take(0)
    sig = signature(toty, fromty)
    res = context.compile_internal(builder, impl, sig, [val])
    return impl_ret_untracked(context, builder, sig.return_type, res)
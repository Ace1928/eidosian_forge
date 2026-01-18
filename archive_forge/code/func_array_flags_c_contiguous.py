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
@lower_getattr(types.ArrayFlags, 'contiguous')
@lower_getattr(types.ArrayFlags, 'c_contiguous')
def array_flags_c_contiguous(context, builder, typ, value):
    if typ.array_type.layout != 'C':
        flagsobj = context.make_helper(builder, typ, value=value)
        res = _call_contiguous_check(is_contiguous, context, builder, typ.array_type, flagsobj.parent)
    else:
        val = typ.array_type.layout == 'C'
        res = context.get_constant(types.boolean, val)
    return impl_ret_untracked(context, builder, typ, res)
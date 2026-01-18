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
@lower_builtin('array.item', types.Array)
def array_item(context, builder, sig, args):
    aryty, = sig.args
    ary, = args
    ary = make_array(aryty)(context, builder, ary)
    nitems = ary.nitems
    with builder.if_then(builder.icmp_signed('!=', nitems, nitems.type(1)), likely=False):
        msg = 'item(): can only convert an array of size 1 to a Python scalar'
        context.call_conv.return_user_exc(builder, ValueError, (msg,))
    return load_item(context, builder, aryty, ary.data)
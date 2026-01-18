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
@lower_builtin(operator.getitem, types.Buffer, types.BaseTuple)
def getitem_array_tuple(context, builder, sig, args):
    """
    Basic or advanced indexing with a tuple.
    """
    aryty, tupty = sig.args
    ary, tup = args
    ary = make_array(aryty)(context, builder, ary)
    index_types = tupty.types
    indices = cgutils.unpack_tuple(builder, tup, count=len(tupty))
    index_types, indices = normalize_indices(context, builder, index_types, indices)
    if any((isinstance(ty, types.Array) for ty in index_types)):
        return fancy_getitem(context, builder, sig, args, aryty, ary, index_types, indices)
    res = _getitem_array_generic(context, builder, sig.return_type, aryty, ary, index_types, indices)
    return impl_ret_borrowed(context, builder, sig.return_type, res)
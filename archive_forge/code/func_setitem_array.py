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
@lower_builtin(operator.setitem, types.Buffer, types.Any, types.Any)
def setitem_array(context, builder, sig, args):
    """
    array[a] = scalar_or_array
    array[a,..,b] = scalar_or_array
    """
    aryty, idxty, valty = sig.args
    ary, idx, val = args
    if isinstance(idxty, types.BaseTuple):
        index_types = idxty.types
        indices = cgutils.unpack_tuple(builder, idx, count=len(idxty))
    else:
        index_types = (idxty,)
        indices = (idx,)
    ary = make_array(aryty)(context, builder, ary)
    index_types, indices = normalize_indices(context, builder, index_types, indices)
    try:
        dataptr, shapes, strides = basic_indexing(context, builder, aryty, ary, index_types, indices, boundscheck=context.enable_boundscheck)
    except NotImplementedError:
        use_fancy_indexing = True
    else:
        use_fancy_indexing = bool(shapes)
    if use_fancy_indexing:
        return fancy_setslice(context, builder, sig, args, index_types, indices)
    val = context.cast(builder, val, valty, aryty.dtype)
    store_item(context, builder, aryty, val, dataptr)
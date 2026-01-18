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
@lower_builtin(operator.getitem, types.Buffer, types.Integer)
@lower_builtin(operator.getitem, types.Buffer, types.SliceType)
def getitem_arraynd_intp(context, builder, sig, args):
    """
    Basic indexing with an integer or a slice.
    """
    aryty, idxty = sig.args
    ary, idx = args
    assert aryty.ndim >= 1
    ary = make_array(aryty)(context, builder, ary)
    res = _getitem_array_generic(context, builder, sig.return_type, aryty, ary, (idxty,), (idx,))
    return impl_ret_borrowed(context, builder, sig.return_type, res)
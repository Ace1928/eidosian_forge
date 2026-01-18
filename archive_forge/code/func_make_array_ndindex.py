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
@lower_builtin(pndindex, types.VarArg(types.Integer))
@lower_builtin(np.ndindex, types.VarArg(types.Integer))
def make_array_ndindex(context, builder, sig, args):
    """ndindex(*shape)"""
    shape = [context.cast(builder, arg, argty, types.intp) for argty, arg in zip(sig.args, args)]
    nditercls = make_ndindex_cls(types.NumpyNdIndexType(len(shape)))
    nditer = nditercls(context, builder)
    nditer.init_specific(context, builder, shape)
    res = nditer._getvalue()
    return impl_ret_borrowed(context, builder, sig.return_type, res)
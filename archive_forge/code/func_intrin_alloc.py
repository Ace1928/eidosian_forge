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
@intrinsic
def intrin_alloc(typingctx, allocsize, align):
    """Intrinsic to call into the allocator for Array
    """

    def codegen(context, builder, signature, args):
        [allocsize, align] = args
        meminfo = context.nrt.meminfo_alloc_aligned(builder, allocsize, align)
        return meminfo
    mip = types.MemInfoPointer(types.voidptr)
    sig = signature(mip, allocsize, align)
    return (sig, codegen)
import operator
import numpy as np
from llvmlite import ir
from numba.core import types, cgutils
from numba.core.extending import (overload, intrinsic, overload_method,
from numba.core.cgutils import is_nonelike
from numba.cpython import unicode
@lower_cast(types.CharSeq, types.Bytes)
def charseq_to_bytes(context, builder, fromty, toty, val):
    bstr = _make_constant_bytes(context, builder, val.type.count)
    rawptr = cgutils.alloca_once_value(builder, value=val)
    ptr = builder.bitcast(rawptr, bstr.data.type)
    cgutils.memcpy(builder, bstr.data, ptr, bstr.nitems)
    return bstr
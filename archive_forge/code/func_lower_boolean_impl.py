import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def lower_boolean_impl(context, builder, sig, args):
    libfunc_impl = context.get_function(libfunc, typing.signature(types.int32, ty))
    result = libfunc_impl(builder, args)
    return context.cast(builder, result, types.int32, types.boolean)
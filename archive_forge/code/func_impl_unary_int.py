import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def impl_unary_int(key, ty, libfunc):

    def lower_unary_int_impl(context, builder, sig, args):
        if sig.args[0] == int64:
            convert = builder.sitofp
        elif sig.args[0] == uint64:
            convert = builder.uitofp
        else:
            m = 'Only 64-bit integers are supported for generic unary int ops'
            raise TypeError(m)
        arg = convert(args[0], ir.DoubleType())
        sig = typing.signature(float64, float64)
        libfunc_impl = context.get_function(libfunc, sig)
        return libfunc_impl(builder, [arg])
    lower(key, ty)(lower_unary_int_impl)
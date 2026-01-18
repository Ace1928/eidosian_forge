import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def impl_pow_int(ty, libfunc):

    def lower_pow_impl_int(context, builder, sig, args):
        powi_sig = typing.signature(ty, ty, types.int32)
        libfunc_impl = context.get_function(libfunc, powi_sig)
        return libfunc_impl(builder, args)
    lower(math.pow, ty, types.int32)(lower_pow_impl_int)
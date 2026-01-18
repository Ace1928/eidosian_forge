import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def impl_ldexp(ty, libfunc):

    def lower_ldexp_impl(context, builder, sig, args):
        ldexp_sig = typing.signature(ty, ty, types.int32)
        libfunc_impl = context.get_function(libfunc, ldexp_sig)
        return libfunc_impl(builder, args)
    lower(math.ldexp, ty, types.int32)(lower_ldexp_impl)
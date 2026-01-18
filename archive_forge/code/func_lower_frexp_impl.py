import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def lower_frexp_impl(context, builder, sig, args):
    frexp_sig = typing.signature(retty, ty)
    libfunc_impl = context.get_function(libfunc, frexp_sig)
    return libfunc_impl(builder, args)
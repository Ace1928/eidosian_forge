import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def tanh_impl_libdevice():
    tanh_sig = typing.signature(ty, ty)
    libfunc_impl = context.get_function(libfunc, tanh_sig)
    return libfunc_impl(builder, args)
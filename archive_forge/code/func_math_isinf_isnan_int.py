import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
@lower(math.isinf, types.Integer)
@lower(math.isnan, types.Integer)
def math_isinf_isnan_int(context, builder, sig, args):
    return context.get_constant(types.boolean, 0)
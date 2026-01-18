import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def tanhf_impl_fastmath():
    fnty = ir.FunctionType(ir.FloatType(), [ir.FloatType()])
    asm = ir.InlineAsm(fnty, 'tanh.approx.f32 $0, $1;', '=f,f')
    return builder.call(asm, args)
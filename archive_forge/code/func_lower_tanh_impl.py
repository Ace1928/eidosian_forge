import math
import operator
from llvmlite import ir
from numba.core import types, typing, cgutils, targetconfig
from numba.core.imputils import Registry
from numba.types import float32, float64, int64, uint64
from numba.cuda import libdevice
from numba import cuda
def lower_tanh_impl(context, builder, sig, args):

    def get_compute_capability():
        flags = targetconfig.ConfigStack().top()
        return flags.compute_capability

    def tanh_impl_libdevice():
        tanh_sig = typing.signature(ty, ty)
        libfunc_impl = context.get_function(libfunc, tanh_sig)
        return libfunc_impl(builder, args)

    def tanhf_impl_fastmath():
        fnty = ir.FunctionType(ir.FloatType(), [ir.FloatType()])
        asm = ir.InlineAsm(fnty, 'tanh.approx.f32 $0, $1;', '=f,f')
        return builder.call(asm, args)
    if ty == float32 and context.fastmath:
        cc = get_compute_capability()
        if cc >= (7, 5):
            return tanhf_impl_fastmath()
    return tanh_impl_libdevice()
import math
import numbers
import numpy as np
import operator
from llvmlite import ir
from llvmlite.ir import Constant
from numba.core.imputils import (lower_builtin, lower_getattr,
from numba.core import typing, types, utils, errors, cgutils, optional
from numba.core.extending import intrinsic, overload_method
from numba.cpython.unsafe.numbers import viewer
@lower_builtin(operator.pow, types.Complex, types.Complex)
@lower_builtin(operator.ipow, types.Complex, types.Complex)
@lower_builtin(pow, types.Complex, types.Complex)
def complex_power_impl(context, builder, sig, args):
    [ca, cb] = args
    ty = sig.args[0]
    fty = ty.underlying_float
    a = context.make_helper(builder, ty, value=ca)
    b = context.make_helper(builder, ty, value=cb)
    c = context.make_helper(builder, ty)
    module = builder.module
    pa = a._getpointer()
    pb = b._getpointer()
    pc = c._getpointer()
    TWO = context.get_constant(fty, 2)
    ZERO = context.get_constant(fty, 0)
    b_real_is_two = builder.fcmp_ordered('==', b.real, TWO)
    b_imag_is_zero = builder.fcmp_ordered('==', b.imag, ZERO)
    b_is_two = builder.and_(b_real_is_two, b_imag_is_zero)
    with builder.if_else(b_is_two) as (then, otherwise):
        with then:
            res = complex_mul_impl(context, builder, sig, (ca, ca))
            cres = context.make_helper(builder, ty, value=res)
            c.real = cres.real
            c.imag = cres.imag
        with otherwise:
            func_name = {types.complex64: 'numba_cpowf', types.complex128: 'numba_cpow'}[ty]
            fnty = ir.FunctionType(ir.VoidType(), [pa.type] * 3)
            cpow = cgutils.get_or_insert_function(module, fnty, func_name)
            builder.call(cpow, (pa, pb, pc))
    res = builder.load(pc)
    return impl_ret_untracked(context, builder, sig.return_type, res)
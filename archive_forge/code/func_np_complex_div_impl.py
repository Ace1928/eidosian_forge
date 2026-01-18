import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_div_impl(context, builder, sig, args):
    in1, in2 = [context.make_complex(builder, sig.args[0], value=arg) for arg in args]
    in1r = in1.real
    in1i = in1.imag
    in2r = in2.real
    in2i = in2.imag
    ftype = in1r.type
    assert all([i.type == ftype for i in [in1r, in1i, in2r, in2i]]), 'mismatched types'
    out = context.make_helper(builder, sig.return_type)
    ZERO = llvmlite.ir.Constant(ftype, 0.0)
    ONE = llvmlite.ir.Constant(ftype, 1.0)
    in2r_abs = _fabs(context, builder, in2r)
    in2i_abs = _fabs(context, builder, in2i)
    in2r_abs_ge_in2i_abs = builder.fcmp_ordered('>=', in2r_abs, in2i_abs)
    with builder.if_else(in2r_abs_ge_in2i_abs) as (then, otherwise):
        with then:
            in2r_is_zero = builder.fcmp_ordered('==', in2r_abs, ZERO)
            in2i_is_zero = builder.fcmp_ordered('==', in2i_abs, ZERO)
            in2_is_zero = builder.and_(in2r_is_zero, in2i_is_zero)
            with builder.if_else(in2_is_zero) as (inn_then, inn_otherwise):
                with inn_then:
                    out.real = builder.fdiv(in1r, in2r_abs)
                    out.imag = builder.fdiv(in1i, in2i_abs)
                with inn_otherwise:
                    rat = builder.fdiv(in2i, in2r)
                    tmp1 = builder.fmul(in2i, rat)
                    tmp2 = builder.fadd(in2r, tmp1)
                    scl = builder.fdiv(ONE, tmp2)
                    tmp3 = builder.fmul(in1i, rat)
                    tmp4 = builder.fmul(in1r, rat)
                    tmp5 = builder.fadd(in1r, tmp3)
                    tmp6 = builder.fsub(in1i, tmp4)
                    out.real = builder.fmul(tmp5, scl)
                    out.imag = builder.fmul(tmp6, scl)
        with otherwise:
            rat = builder.fdiv(in2r, in2i)
            tmp1 = builder.fmul(in2r, rat)
            tmp2 = builder.fadd(in2i, tmp1)
            scl = builder.fdiv(ONE, tmp2)
            tmp3 = builder.fmul(in1r, rat)
            tmp4 = builder.fmul(in1i, rat)
            tmp5 = builder.fadd(tmp3, in1i)
            tmp6 = builder.fsub(tmp4, in1r)
            out.real = builder.fmul(tmp5, scl)
            out.imag = builder.fmul(tmp6, scl)
    return out._getvalue()
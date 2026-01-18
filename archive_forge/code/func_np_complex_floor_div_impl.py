import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_complex_floor_div_impl(context, builder, sig, args):
    float_kind = sig.args[0].underlying_float
    floor_sig = typing.signature(float_kind, float_kind)
    in1, in2 = [context.make_complex(builder, sig.args[0], value=arg) for arg in args]
    in1r = in1.real
    in1i = in1.imag
    in2r = in2.real
    in2i = in2.imag
    ftype = in1r.type
    assert all([i.type == ftype for i in [in1r, in1i, in2r, in2i]]), 'mismatched types'
    ZERO = llvmlite.ir.Constant(ftype, 0.0)
    out = context.make_helper(builder, sig.return_type)
    out.imag = ZERO
    in2r_abs = _fabs(context, builder, in2r)
    in2i_abs = _fabs(context, builder, in2i)
    in2r_abs_ge_in2i_abs = builder.fcmp_ordered('>=', in2r_abs, in2i_abs)
    with builder.if_else(in2r_abs_ge_in2i_abs) as (then, otherwise):
        with then:
            rat = builder.fdiv(in2i, in2r)
            tmp1 = builder.fmul(in1i, rat)
            tmp2 = builder.fmul(in2i, rat)
            tmp3 = builder.fadd(in1r, tmp1)
            tmp4 = builder.fadd(in2r, tmp2)
            tmp5 = builder.fdiv(tmp3, tmp4)
            out.real = np_real_floor_impl(context, builder, floor_sig, (tmp5,))
        with otherwise:
            rat = builder.fdiv(in2r, in2i)
            tmp1 = builder.fmul(in1r, rat)
            tmp2 = builder.fmul(in2r, rat)
            tmp3 = builder.fadd(in1i, tmp1)
            tmp4 = builder.fadd(in2i, tmp2)
            tmp5 = builder.fdiv(tmp3, tmp4)
            out.real = np_real_floor_impl(context, builder, floor_sig, (tmp5,))
    return out._getvalue()
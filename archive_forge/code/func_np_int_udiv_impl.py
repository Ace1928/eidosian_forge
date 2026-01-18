import math
import llvmlite.ir
import numpy as np
from numba.core.extending import overload
from numba.core.imputils import impl_ret_untracked
from numba.core import typing, types, errors, lowering, cgutils
from numba.core.extending import register_jitable
from numba.np import npdatetime
from numba.cpython import cmathimpl, mathimpl, numbers
def np_int_udiv_impl(context, builder, sig, args):
    _check_arity_and_homogeneity(sig, args, 2)
    num, den = args
    ty = sig.args[0]
    ZERO = context.get_constant(ty, 0)
    div_by_zero = builder.icmp_unsigned('==', ZERO, den)
    with builder.if_else(div_by_zero, likely=False) as (then, otherwise):
        with then:
            bb_then = builder.basic_block
        with otherwise:
            div = builder.udiv(num, den)
            bb_otherwise = builder.basic_block
    result = builder.phi(ZERO.type)
    result.add_incoming(ZERO, bb_then)
    result.add_incoming(div, bb_otherwise)
    return result
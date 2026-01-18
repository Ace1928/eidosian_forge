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
@lower_builtin(operator.pow, types.Integer, types.IntegerLiteral)
@lower_builtin(operator.ipow, types.Integer, types.IntegerLiteral)
@lower_builtin(operator.pow, types.Float, types.IntegerLiteral)
@lower_builtin(operator.ipow, types.Float, types.IntegerLiteral)
def static_power_impl(context, builder, sig, args):
    """
    a ^ b, where a is an integer or real, and b a constant integer
    """
    exp = sig.args[1].value
    if not isinstance(exp, numbers.Integral):
        raise NotImplementedError
    if abs(exp) > 65536:
        raise NotImplementedError
    invert = exp < 0
    exp = abs(exp)
    tp = sig.return_type
    is_integer = isinstance(tp, types.Integer)
    zerodiv_return = _get_power_zerodiv_return(context, tp)
    val = context.cast(builder, args[0], sig.args[0], tp)
    lty = val.type

    def mul(a, b):
        if is_integer:
            return builder.mul(a, b)
        else:
            return builder.fmul(a, b)
    res = lty(1)
    a = val
    while exp != 0:
        if exp & 1:
            res = mul(res, val)
        exp >>= 1
        val = mul(val, val)
    if invert:
        if is_integer:

            def invert_impl(a):
                if a == 0:
                    if zerodiv_return:
                        return zerodiv_return
                    else:
                        raise ZeroDivisionError('0 cannot be raised to a negative power')
                if a != 1 and a != -1:
                    return 0
                else:
                    return a
        else:

            def invert_impl(a):
                return 1.0 / a
        res = context.compile_internal(builder, invert_impl, typing.signature(tp, tp), (res,))
    return res
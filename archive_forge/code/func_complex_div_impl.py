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
def complex_div_impl(context, builder, sig, args):

    def complex_div(a, b):
        areal = a.real
        aimag = a.imag
        breal = b.real
        bimag = b.imag
        if not breal and (not bimag):
            raise ZeroDivisionError('complex division by zero')
        if abs(breal) >= abs(bimag):
            if not breal:
                return complex(NAN, NAN)
            ratio = bimag / breal
            denom = breal + bimag * ratio
            return complex((areal + aimag * ratio) / denom, (aimag - areal * ratio) / denom)
        else:
            if not bimag:
                return complex(NAN, NAN)
            ratio = breal / bimag
            denom = breal * ratio + bimag
            return complex((a.real * ratio + a.imag) / denom, (a.imag * ratio - a.real) / denom)
    res = context.compile_internal(builder, complex_div, sig, args)
    return impl_ret_untracked(context, builder, sig.return_type, res)
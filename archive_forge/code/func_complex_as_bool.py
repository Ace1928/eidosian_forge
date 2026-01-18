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
@lower_builtin(bool, types.Complex)
def complex_as_bool(context, builder, sig, args):
    [typ] = sig.args
    [val] = args
    cmplx = context.make_complex(builder, typ, val)
    real, imag = (cmplx.real, cmplx.imag)
    zero = Constant(real.type, 0.0)
    real_istrue = builder.fcmp_unordered('!=', real, zero)
    imag_istrue = builder.fcmp_unordered('!=', imag, zero)
    return builder.or_(real_istrue, imag_istrue)
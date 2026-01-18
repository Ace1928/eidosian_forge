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
@lower_cast(types.Float, types.Complex)
@lower_cast(types.Integer, types.Complex)
def non_complex_to_complex(context, builder, fromty, toty, val):
    real = context.cast(builder, val, fromty, toty.underlying_float)
    imag = context.get_constant(toty.underlying_float, 0)
    cmplx = context.make_complex(builder, toty)
    cmplx.real = real
    cmplx.imag = imag
    return cmplx._getvalue()
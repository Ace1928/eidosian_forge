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
@lower_constant(types.Complex)
def constant_complex(context, builder, ty, pyval):
    fty = ty.underlying_float
    real = context.get_constant_generic(builder, fty, pyval.real)
    imag = context.get_constant_generic(builder, fty, pyval.imag)
    return Constant.literal_struct((real, imag))
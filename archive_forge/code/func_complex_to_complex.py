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
@lower_cast(types.Complex, types.Complex)
def complex_to_complex(context, builder, fromty, toty, val):
    srcty = fromty.underlying_float
    dstty = toty.underlying_float
    src = context.make_complex(builder, fromty, value=val)
    dst = context.make_complex(builder, toty)
    dst.real = context.cast(builder, src.real, srcty, dstty)
    dst.imag = context.cast(builder, src.imag, srcty, dstty)
    return dst._getvalue()
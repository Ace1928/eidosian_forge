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
@lower_cast(types.Integer, types.Integer)
def integer_to_integer(context, builder, fromty, toty, val):
    if toty.bitwidth == fromty.bitwidth:
        return val
    elif toty.bitwidth < fromty.bitwidth:
        return builder.trunc(val, context.get_value_type(toty))
    elif fromty.signed:
        return builder.sext(val, context.get_value_type(toty))
    else:
        return builder.zext(val, context.get_value_type(toty))
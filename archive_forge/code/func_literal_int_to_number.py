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
@lower_cast(types.IntegerLiteral, types.Integer)
@lower_cast(types.IntegerLiteral, types.Float)
@lower_cast(types.IntegerLiteral, types.Complex)
def literal_int_to_number(context, builder, fromty, toty, val):
    lit = context.get_constant_generic(builder, fromty.literal_type, fromty.literal_value)
    return context.cast(builder, lit, fromty.literal_type, toty)
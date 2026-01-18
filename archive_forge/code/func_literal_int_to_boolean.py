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
@lower_cast(types.IntegerLiteral, types.Boolean)
@lower_cast(types.BooleanLiteral, types.Boolean)
def literal_int_to_boolean(context, builder, fromty, toty, val):
    lit = context.get_constant_generic(builder, fromty.literal_type, fromty.literal_value)
    return context.is_true(builder, fromty.literal_type, lit)
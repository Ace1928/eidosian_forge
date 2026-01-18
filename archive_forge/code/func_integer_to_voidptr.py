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
@lower_cast(types.Integer, types.voidptr)
def integer_to_voidptr(context, builder, fromty, toty, val):
    return builder.inttoptr(val, context.get_value_type(toty))
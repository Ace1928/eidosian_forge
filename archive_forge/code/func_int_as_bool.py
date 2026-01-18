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
@lower_builtin(bool, types.Integer)
def int_as_bool(context, builder, sig, args):
    [val] = args
    return builder.icmp_unsigned('!=', val, Constant(val.type, 0))
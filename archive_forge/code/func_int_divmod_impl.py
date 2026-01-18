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
@lower_builtin(divmod, types.Integer, types.Integer)
def int_divmod_impl(context, builder, sig, args):
    quot, rem = _int_divmod_impl(context, builder, sig, args, 'integer divmod by zero')
    return cgutils.pack_array(builder, (builder.load(quot), builder.load(rem)))
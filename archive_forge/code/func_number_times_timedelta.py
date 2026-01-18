import numpy as np
import operator
import llvmlite.ir
from llvmlite.ir import Constant
from numba.core import types, cgutils
from numba.core.cgutils import create_constant_array
from numba.core.imputils import (lower_builtin, lower_constant,
from numba.np import npdatetime_helpers, numpy_support, npyfuncs
from numba.extending import overload_method
from numba.core.config import IS_32BITS
from numba.core.errors import LoweringError
@lower_builtin(operator.mul, types.Integer, types.NPTimedelta)
@lower_builtin(operator.imul, types.Integer, types.NPTimedelta)
@lower_builtin(operator.mul, types.Float, types.NPTimedelta)
@lower_builtin(operator.imul, types.Float, types.NPTimedelta)
def number_times_timedelta(context, builder, sig, args):
    res = _timedelta_times_number(context, builder, args[1], sig.args[1], args[0], sig.args[0], sig.return_type)
    return impl_ret_untracked(context, builder, sig.return_type, res)
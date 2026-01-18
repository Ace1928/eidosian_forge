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
@lower_builtin(operator.add, types.NPDatetime, types.NPTimedelta)
@lower_builtin(operator.iadd, types.NPDatetime, types.NPTimedelta)
def datetime_plus_timedelta(context, builder, sig, args):
    dt_arg, td_arg = args
    dt_type, td_type = sig.args
    res = _datetime_plus_timedelta(context, builder, dt_arg, dt_type.unit, td_arg, td_type.unit, sig.return_type.unit)
    return impl_ret_untracked(context, builder, sig.return_type, res)
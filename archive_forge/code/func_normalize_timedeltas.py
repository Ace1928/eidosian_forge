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
def normalize_timedeltas(context, builder, left, right, leftty, rightty):
    """
    Scale either *left* or *right* to the other's unit, in order to have
    homogeneous units.
    """
    factor = npdatetime_helpers.get_timedelta_conversion_factor(leftty.unit, rightty.unit)
    if factor is not None:
        return (scale_by_constant(builder, left, factor), right)
    factor = npdatetime_helpers.get_timedelta_conversion_factor(rightty.unit, leftty.unit)
    if factor is not None:
        return (left, scale_by_constant(builder, right, factor))
    raise RuntimeError('cannot normalize %r and %r' % (leftty, rightty))
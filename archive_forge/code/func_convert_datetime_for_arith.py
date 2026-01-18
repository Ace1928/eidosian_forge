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
def convert_datetime_for_arith(builder, dt_val, src_unit, dest_unit):
    """
    Convert datetime *dt_val* from *src_unit* to *dest_unit*.
    """
    dt_val, dt_unit = reduce_datetime_for_unit(builder, dt_val, src_unit, dest_unit)
    dt_factor = npdatetime_helpers.get_timedelta_conversion_factor(dt_unit, dest_unit)
    if dt_factor is None:
        raise LoweringError('cannot convert datetime64 from %r to %r' % (src_unit, dest_unit))
    return scale_by_constant(builder, dt_val, dt_factor)
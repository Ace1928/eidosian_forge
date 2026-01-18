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
def scale_timedelta(context, builder, val, srcty, destty):
    """
    Scale the timedelta64 *val* from *srcty* to *destty*
    (both numba.types.NPTimedelta instances)
    """
    factor = npdatetime_helpers.get_timedelta_conversion_factor(srcty.unit, destty.unit)
    if factor is None:
        msg = f'cannot convert timedelta64 from {srcty.unit} to {destty.unit}'
        raise LoweringError(msg)
    return scale_by_constant(builder, val, factor)
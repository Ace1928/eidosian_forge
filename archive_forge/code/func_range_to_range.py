import operator
from numba import prange
from numba.core import types, cgutils, errors
from numba.cpython.listobj import ListIterInstance
from numba.np.arrayobj import make_array
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, overload_attribute, register_jitable
from numba.parfors.parfor import internal_prange
@lower_cast(types.RangeType, types.RangeType)
def range_to_range(context, builder, fromty, toty, val):
    olditems = cgutils.unpack_tuple(builder, val, 3)
    items = [context.cast(builder, v, fromty.dtype, toty.dtype) for v in olditems]
    return cgutils.make_anonymous_struct(builder, items)
import operator
from numba import prange
from numba.core import types, cgutils, errors
from numba.cpython.listobj import ListIterInstance
from numba.np.arrayobj import make_array
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, overload_attribute, register_jitable
from numba.parfors.parfor import internal_prange
@overload(operator.contains)
def impl_contains(robj, val):

    def impl_false(robj, val):
        return False
    if not isinstance(robj, types.RangeType):
        return
    elif isinstance(val, (types.Integer, types.Boolean)):
        return impl_contains_helper
    elif isinstance(val, types.Float):

        def impl(robj, val):
            if val % 1 != 0:
                return False
            else:
                return impl_contains_helper(robj, int(val))
        return impl
    elif isinstance(val, types.Complex):

        def impl(robj, val):
            if val.imag != 0:
                return False
            elif val.real % 1 != 0:
                return False
            else:
                return impl_contains_helper(robj, int(val.real))
        return impl
    elif not isinstance(val, types.Number):
        return impl_false
import operator
from numba import prange
from numba.core import types, cgutils, errors
from numba.cpython.listobj import ListIterInstance
from numba.np.arrayobj import make_array
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, overload_attribute, register_jitable
from numba.parfors.parfor import internal_prange
@register_jitable
def impl_contains_helper(robj, val):
    if robj.step > 0 and (val < robj.start or val >= robj.stop):
        return False
    elif robj.step < 0 and (val <= robj.stop or val > robj.start):
        return False
    return (val - robj.start) % robj.step == 0
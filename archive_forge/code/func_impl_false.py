import operator
from numba import prange
from numba.core import types, cgutils, errors
from numba.cpython.listobj import ListIterInstance
from numba.np.arrayobj import make_array
from numba.core.imputils import (lower_builtin, lower_cast,
from numba.core.typing import signature
from numba.core.extending import intrinsic, overload, overload_attribute, register_jitable
from numba.parfors.parfor import internal_prange
def impl_false(robj, val):
    return False
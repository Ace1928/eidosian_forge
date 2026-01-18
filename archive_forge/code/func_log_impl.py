import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
@lower(cmath.log, types.Complex)
@intrinsic_complex_unary
def log_impl(x, y, x_is_finite, y_is_finite):
    """cmath.log(x + y j)"""
    a = math.log(math.hypot(x, y))
    b = math.atan2(y, x)
    return complex(a, b)
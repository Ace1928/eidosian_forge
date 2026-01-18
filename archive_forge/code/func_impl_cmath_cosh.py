import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
@overload(cmath.cosh)
def impl_cmath_cosh(z):
    if not isinstance(z, types.Complex):
        return

    def cosh_impl(z):
        """cmath.cosh(z)"""
        x = z.real
        y = z.imag
        if math.isinf(x):
            if math.isnan(y):
                real = abs(x)
                imag = y
            elif y == 0.0:
                real = abs(x)
                imag = y
            else:
                real = math.copysign(x, math.cos(y))
                imag = math.copysign(x, math.sin(y))
            if x < 0.0:
                imag = -imag
            return complex(real, imag)
        return complex(math.cos(y) * math.cosh(x), math.sin(y) * math.sinh(x))
    return cosh_impl
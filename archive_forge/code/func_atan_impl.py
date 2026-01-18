import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def atan_impl(z):
    """cmath.atan(z) = -j * cmath.atanh(z j)"""
    r = cmath.atanh(complex(-z.imag, z.real))
    if math.isinf(z.real) and math.isnan(z.imag):
        return complex(r.imag, r.real)
    else:
        return complex(r.imag, -r.real)
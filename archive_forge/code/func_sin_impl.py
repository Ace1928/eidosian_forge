import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def sin_impl(z):
    """cmath.sin(z) = -j * cmath.sinh(z j)"""
    r = cmath.sinh(complex(-z.imag, z.real))
    return complex(r.imag, -r.real)
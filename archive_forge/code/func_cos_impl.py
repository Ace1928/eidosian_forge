import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def cos_impl(z):
    """cmath.cos(z) = cmath.cosh(z j)"""
    return cmath.cosh(complex(-z.imag, z.real))
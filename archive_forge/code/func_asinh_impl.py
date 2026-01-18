import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def asinh_impl(z):
    """cmath.asinh(z)"""
    if abs(z.real) > THRES or abs(z.imag) > THRES:
        real = math.copysign(math.log(math.hypot(z.real * 0.5, z.imag * 0.5)) + LN_4, z.real)
        imag = math.atan2(z.imag, abs(z.real))
        return complex(real, imag)
    else:
        s1 = cmath.sqrt(complex(1.0 + z.imag, -z.real))
        s2 = cmath.sqrt(complex(1.0 - z.imag, z.real))
        real = math.asinh(s1.real * s2.imag - s2.real * s1.imag)
        imag = math.atan2(z.imag, s1.real * s2.real - s1.imag * s2.imag)
        return complex(real, imag)
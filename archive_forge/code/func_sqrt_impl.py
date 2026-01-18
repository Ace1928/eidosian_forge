import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def sqrt_impl(z):
    """cmath.sqrt(z)"""
    a = z.real
    b = z.imag
    if a == 0.0 and b == 0.0:
        return complex(abs(b), b)
    if math.isinf(b):
        return complex(abs(b), b)
    if math.isnan(a):
        return complex(a, a)
    if math.isinf(a):
        if a < 0.0:
            return complex(abs(b - b), math.copysign(a, b))
        else:
            return complex(a, math.copysign(b - b, b))
    if abs(a) >= THRES or abs(b) >= THRES:
        a *= 0.25
        b *= 0.25
        scale = True
    else:
        scale = False
    if a >= 0:
        t = math.sqrt((a + math.hypot(a, b)) * 0.5)
        real = t
        imag = b / (2 * t)
    else:
        t = math.sqrt((-a + math.hypot(a, b)) * 0.5)
        real = abs(b) / (2 * t)
        imag = math.copysign(t, b)
    if scale:
        return complex(real * 2, imag)
    else:
        return complex(real, imag)
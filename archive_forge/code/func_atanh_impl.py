import cmath
import math
from numba.core.imputils import Registry, impl_ret_untracked
from numba.core import types, cgutils
from numba.core.typing import signature
from numba.cpython import builtins, mathimpl
from numba.core.extending import overload
def atanh_impl(z):
    """cmath.atanh(z)"""
    if z.real < 0.0:
        negate = True
        z = -z
    else:
        negate = False
    ay = abs(z.imag)
    if math.isnan(z.real) or z.real > THRES_LARGE or ay > THRES_LARGE:
        if math.isinf(z.imag):
            real = math.copysign(0.0, z.real)
        elif math.isinf(z.real):
            real = 0.0
        else:
            h = math.hypot(z.real * 0.5, z.imag * 0.5)
            real = z.real / 4.0 / h / h
        imag = -math.copysign(PI_12, -z.imag)
    elif z.real == 1.0 and ay < THRES_SMALL:
        if ay == 0.0:
            real = INF
            imag = z.imag
        else:
            real = -math.log(math.sqrt(ay) / math.sqrt(math.hypot(ay, 2.0)))
            imag = math.copysign(math.atan2(2.0, -ay) / 2, z.imag)
    else:
        sqay = ay * ay
        zr1 = 1 - z.real
        real = math.log1p(4.0 * z.real / (zr1 * zr1 + sqay)) * 0.25
        imag = -math.atan2(-2.0 * z.imag, zr1 * (1 + z.real) - sqay) * 0.5
    if math.isnan(z.imag):
        imag = NAN
    if negate:
        return complex(-real, -imag)
    else:
        return complex(real, imag)
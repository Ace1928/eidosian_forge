import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_triangular(bitgen, left, mode, right):
    base = right - left
    leftbase = mode - left
    ratio = leftbase / base
    leftprod = leftbase * base
    rightprod = (right - mode) * base
    U = next_double(bitgen)
    if U <= ratio:
        return left + np.sqrt(U * leftprod)
    else:
        return right - np.sqrt((1.0 - U) * rightprod)
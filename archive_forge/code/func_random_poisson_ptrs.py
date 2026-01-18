import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_poisson_ptrs(bitgen, lam):
    slam = np.sqrt(lam)
    loglam = np.log(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2)
    while 1:
        U = next_double(bitgen) - 0.5
        V = next_double(bitgen)
        us = 0.5 - np.fabs(U)
        k = int((2 * a / us + b) * U + lam + 0.43)
        if us >= 0.07 and V <= vr:
            return k
        if k < 0 or (us < 0.013 and V > us):
            continue
        if np.log(V) + np.log(invalpha) - np.log(a / (us * us) + b) <= -lam + k * loglam - random_loggam(k + 1):
            return k
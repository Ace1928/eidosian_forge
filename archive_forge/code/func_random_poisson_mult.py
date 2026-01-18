import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_poisson_mult(bitgen, lam):
    enlam = np.exp(-lam)
    X = 0
    prod = 1.0
    while 1:
        U = next_double(bitgen)
        prod *= U
        if prod > enlam:
            X += 1
        else:
            return X
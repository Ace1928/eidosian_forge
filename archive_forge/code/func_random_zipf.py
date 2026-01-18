import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_zipf(bitgen, a):
    am1 = a - 1.0
    b = pow(2.0, am1)
    while 1:
        U = 1.0 - next_double(bitgen)
        V = next_double(bitgen)
        X = np.floor(pow(U, -1.0 / am1))
        if X > INT64_MAX or X < 1.0:
            continue
        T = pow(1.0 + 1.0 / X, am1)
        if V * X * (T - 1.0) / (b - 1.0) <= T / b:
            return X
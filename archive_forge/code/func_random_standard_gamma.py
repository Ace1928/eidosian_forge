import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_standard_gamma(bitgen, shape):
    if shape == 1.0:
        return random_standard_exponential(bitgen)
    elif shape == 0.0:
        return 0.0
    elif shape < 1.0:
        while 1:
            U = next_double(bitgen)
            V = random_standard_exponential(bitgen)
            if U <= 1.0 - shape:
                X = pow(U, 1.0 / shape)
                if X <= V:
                    return X
            else:
                Y = -np.log((1 - U) / shape)
                X = pow(1.0 - shape + shape * Y, 1.0 / shape)
                if X <= V + Y:
                    return X
    else:
        b = shape - 1.0 / 3.0
        c = 1.0 / np.sqrt(9 * b)
        while 1:
            while 1:
                X = random_standard_normal(bitgen)
                V = 1.0 + c * X
                if V > 0.0:
                    break
            V = V * V * V
            U = next_double(bitgen)
            if U < 1.0 - 0.0331 * (X * X) * (X * X):
                return b * V
            if np.log(U) < 0.5 * X * X + b * (1.0 - V + np.log(V)):
                return b * V
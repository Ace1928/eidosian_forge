import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_standard_gamma_f(bitgen, shape):
    f32_one = float32(1.0)
    shape = float32(shape)
    if shape == f32_one:
        return random_standard_exponential_f(bitgen)
    elif shape == float32(0.0):
        return float32(0.0)
    elif shape < f32_one:
        while 1:
            U = next_float(bitgen)
            V = random_standard_exponential_f(bitgen)
            if U <= f32_one - shape:
                X = float32(pow(U, float32(f32_one / shape)))
                if X <= V:
                    return X
            else:
                Y = float32(-np.log(float32((f32_one - U) / shape)))
                X = float32(pow(f32_one - shape + float32(shape * Y), float32(f32_one / shape)))
                if X <= V + Y:
                    return X
    else:
        b = shape - f32_one / float32(3.0)
        c = float32(f32_one / float32(np.sqrt(float32(9.0) * b)))
        while 1:
            while 1:
                X = float32(random_standard_normal_f(bitgen))
                V = float32(f32_one + c * X)
                if V > float32(0.0):
                    break
            V = float32(V * V * V)
            U = next_float(bitgen)
            if U < f32_one - float32(0.0331) * (X * X) * (X * X):
                return float32(b * V)
            if np.log(U) < float32(0.5) * X * X + b * (f32_one - V + np.log(V)):
                return float32(b * V)
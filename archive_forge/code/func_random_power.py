import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_power(bitgen, a):
    return pow(-np_expm1(-random_standard_exponential(bitgen)), 1.0 / a)
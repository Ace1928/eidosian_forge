import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_standard_exponential_inv_f(bitgen):
    return -np.log(float32(1.0) - next_float(bitgen))
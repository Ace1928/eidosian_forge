import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_standard_t(bitgen, df):
    num = random_standard_normal(bitgen)
    denom = random_standard_gamma(bitgen, df / 2)
    return np.sqrt(df / 2) * num / np.sqrt(denom)
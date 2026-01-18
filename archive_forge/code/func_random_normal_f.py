import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_normal_f(bitgen, loc, scale):
    scaled_normal = float32(scale * random_standard_normal_f(bitgen))
    return float32(loc + scaled_normal)
import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_geometric(bitgen, p):
    if p >= 0.3333333333333333:
        return random_geometric_search(bitgen, p)
    else:
        return random_geometric_inversion(bitgen, p)
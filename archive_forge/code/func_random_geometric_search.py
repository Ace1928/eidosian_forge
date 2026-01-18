import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_geometric_search(bitgen, p):
    X = 1
    sum = prod = p
    q = 1.0 - p
    U = next_double(bitgen)
    while U > sum:
        prod *= q
        sum += prod
        X = X + 1
    return X
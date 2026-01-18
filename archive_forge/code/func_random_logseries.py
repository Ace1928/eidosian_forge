import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_logseries(bitgen, p):
    r = np_log1p(-p)
    while 1:
        V = next_double(bitgen)
        if V >= p:
            return 1
        U = next_double(bitgen)
        q = -np.expm1(r * U)
        if V <= q * q:
            result = int64(np.floor(1 + np.log(V) / np.log(q)))
            if result < 1 or V == 0.0:
                continue
            else:
                return result
        if V >= q:
            return 1
        else:
            return 2
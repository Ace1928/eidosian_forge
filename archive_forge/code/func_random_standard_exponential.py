import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_standard_exponential(bitgen):
    while 1:
        ri = next_uint64(bitgen)
        ri >>= 3
        idx = ri & 255
        ri >>= 8
        x = ri * we_double[idx]
        if ri < ke_double[idx]:
            return x
        elif idx == 0:
            return ziggurat_exp_r - np_log1p(-next_double(bitgen))
        elif (fe_double[idx - 1] - fe_double[idx]) * next_double(bitgen) + fe_double[idx] < np.exp(-x):
            return x
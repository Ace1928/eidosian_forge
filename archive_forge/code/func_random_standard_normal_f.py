import numpy as np
from numba.core.extending import register_jitable
from numba.np.random._constants import (wi_double, ki_double,
from numba.np.random.generator_core import (next_double, next_float,
from numba import float32, int64
@register_jitable
def random_standard_normal_f(bitgen):
    while 1:
        r = next_uint32(bitgen)
        idx = r & 255
        sign = r >> 8 & 1
        rabs = r >> 9 & 8388607
        x = float32(float32(rabs) * wi_float[idx])
        if sign & 1:
            x = -x
        if rabs < ki_float[idx]:
            return x
        if idx == 0:
            while 1:
                xx = float32(-ziggurat_nor_inv_r_f * np_log1pf(-next_float(bitgen)))
                yy = float32(-np_log1pf(-next_float(bitgen)))
                if float32(yy + yy) > float32(xx * xx):
                    if rabs >> 8 & 1:
                        return -float32(ziggurat_nor_r_f + xx)
                    else:
                        return float32(ziggurat_nor_r_f + xx)
        elif (fi_float[idx - 1] - fi_float[idx]) * next_float(bitgen) + fi_float[idx] < float32(np.exp(-float32(0.5) * x * x)):
            return x
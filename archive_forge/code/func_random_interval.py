import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64
@register_jitable
def random_interval(bitgen, max_val):
    if max_val == 0:
        return 0
    max_val = uint64(max_val)
    mask = uint64(gen_mask(max_val))
    if max_val <= 4294967295:
        value = uint64(next_uint32(bitgen)) & mask
        while value > max_val:
            value = uint64(next_uint32(bitgen)) & mask
    else:
        value = next_uint64(bitgen) & mask
        while value > max_val:
            value = next_uint64(bitgen) & mask
    return uint64(value)
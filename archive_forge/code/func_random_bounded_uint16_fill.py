import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64
@register_jitable
def random_bounded_uint16_fill(bitgen, low, rng, size, dtype):
    """
    Returns a new array of given size with 16 bit integers
    bounded by given interval.
    """
    buf = 0
    bcnt = 0
    out = np.empty(size, dtype=dtype)
    if rng == 0:
        for i in np.ndindex(size):
            out[i] = low
    elif rng == 65535:
        for i in np.ndindex(size):
            val, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)
            out[i] = low + val
    else:
        for i in np.ndindex(size):
            val, bcnt, buf = buffered_bounded_lemire_uint16(bitgen, rng, bcnt, buf)
            out[i] = low + val
    return out
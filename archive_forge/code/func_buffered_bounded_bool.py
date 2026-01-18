import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64
@register_jitable
def buffered_bounded_bool(bitgen, off, rng, bcnt, buf):
    if rng == 0:
        return (off, bcnt, buf)
    if not bcnt:
        buf = next_uint32(bitgen)
        bcnt = 31
    else:
        buf >>= 1
        bcnt -= 1
    return (buf & 1 != 0, bcnt, buf)
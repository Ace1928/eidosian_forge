import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64
@register_jitable
def buffered_uint16(bitgen, bcnt, buf):
    if not bcnt:
        buf = next_uint32(bitgen)
        bcnt = 1
    else:
        buf >>= 16
        bcnt -= 1
    return (uint16(buf), bcnt, buf)
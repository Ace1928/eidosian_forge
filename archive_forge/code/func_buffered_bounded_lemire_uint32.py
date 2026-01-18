import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64
@register_jitable
def buffered_bounded_lemire_uint32(bitgen, rng):
    """
    Generates a random unsigned 32 bit integer bounded
    within a given interval using Lemire's rejection.
    """
    rng_excl = uint32(rng) + uint32(1)
    assert rng != 4294967295
    m = uint64(next_uint32(bitgen)) * uint64(rng_excl)
    leftover = m & 4294967295
    if leftover < rng_excl:
        threshold = (UINT32_MAX - rng) % rng_excl
        while leftover < threshold:
            m = uint64(next_uint32(bitgen)) * uint64(rng_excl)
            leftover = m & 4294967295
    return m >> 32
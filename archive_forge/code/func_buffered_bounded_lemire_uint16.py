import numpy as np
from numba import uint64, uint32, uint16, uint8
from numba.core.extending import register_jitable
from numba.np.random._constants import (UINT32_MAX, UINT64_MAX,
from numba.np.random.generator_core import next_uint32, next_uint64
@register_jitable
def buffered_bounded_lemire_uint16(bitgen, rng, bcnt, buf):
    """
    Generates a random unsigned 16 bit integer bounded
    within a given interval using Lemire's rejection.

    The buffer acts as storage for a 32 bit integer
    drawn from the associated BitGenerator so that
    multiple integers of smaller bitsize can be generated
    from a single draw of the BitGenerator.
    """
    rng_excl = uint16(rng) + uint16(1)
    assert rng != 65535
    n, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)
    m = uint32(n * rng_excl)
    leftover = m & 65535
    if leftover < rng_excl:
        threshold = (uint16(UINT16_MAX) - rng) % rng_excl
        while leftover < threshold:
            n, bcnt, buf = buffered_uint16(bitgen, bcnt, buf)
            m = uint32(n * rng_excl)
            leftover = m & 65535
    return (m >> 16, bcnt, buf)
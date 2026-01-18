import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
@jit(forceobj=_forceobj, looplift=_looplift, nopython=_nopython)
def init_xoroshiro128p_state(states, index, seed):
    """Use SplitMix64 to generate an xoroshiro128p state from 64-bit seed.

    This ensures that manually set small seeds don't result in a predictable
    initial sequence from the random number generator.

    :type states: 1D array, dtype=xoroshiro128p_dtype
    :param states: array of RNG states
    :type index: uint64
    :param index: offset in states to update
    :type seed: int64
    :param seed: seed value to use when initializing state
    """
    index = int64(index)
    seed = uint64(seed)
    z = seed + uint64(11400714819323198485)
    z = (z ^ z >> uint32(30)) * uint64(13787848793156543929)
    z = (z ^ z >> uint32(27)) * uint64(10723151780598845931)
    z = z ^ z >> uint32(31)
    states[index]['s0'] = z
    states[index]['s1'] = z
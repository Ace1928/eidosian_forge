import math
from numba import (config, cuda, float32, float64, uint32, int64, uint64,
import numpy as np
def create_xoroshiro128p_states(n, seed, subsequence_start=0, stream=0):
    """Returns a new device array initialized for n random number generators.

    This initializes the RNG states so that each state in the array corresponds
    subsequences in the separated by 2**64 steps from each other in the main
    sequence.  Therefore, as long no CUDA thread requests more than 2**64
    random numbers, all of the RNG states produced by this function are
    guaranteed to be independent.

    The subsequence_start parameter can be used to advance the first RNG state
    by a multiple of 2**64 steps.

    :type n: int
    :param n: number of RNG states to create
    :type seed: uint64
    :param seed: starting seed for list of generators
    :type subsequence_start: uint64
    :param subsequence_start:
    :type stream: CUDA stream
    :param stream: stream to run initialization kernel on
    """
    states = cuda.device_array(n, dtype=xoroshiro128p_dtype, stream=stream)
    init_xoroshiro128p_states(states, seed, subsequence_start, stream)
    return states
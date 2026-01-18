import numpy as np
from collections import namedtuple
def make_jit_mergesort(*args, **kwargs):
    from numba import njit
    return make_mergesort_impl(njit, *args, **kwargs)
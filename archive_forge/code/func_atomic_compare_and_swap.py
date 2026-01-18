import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def atomic_compare_and_swap(res, old, ary, fill_val):
    gid = cuda.grid(1)
    if gid < res.size:
        old[gid] = cuda.atomic.compare_and_swap(res[gid:], fill_val, ary[gid])
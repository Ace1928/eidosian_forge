import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def atomic_add_double_global_wrap(idx, ary):
    atomic_binary_1dim_global(ary, idx, 32, 1.0, cuda.atomic.add, True)
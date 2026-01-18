import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def atomic_sub_double_global_3(ary):
    atomic_binary_2dim_shared(ary, 1.0, float64, (4, 8), cuda.atomic.sub, atomic_cast_to_uint64, False)
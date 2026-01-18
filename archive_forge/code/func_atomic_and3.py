import numpy as np
from textwrap import dedent
from numba import cuda, uint32, uint64, float32, float64
from numba.cuda.testing import unittest, CUDATestCase, cc_X_or_above
from numba.core import config
def atomic_and3(ary, op2):
    atomic_binary_2dim_shared(ary, op2, uint32, (4, 8), cuda.atomic.and_, atomic_cast_to_uint64, False)
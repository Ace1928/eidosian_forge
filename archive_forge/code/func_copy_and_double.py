import numpy as np
from collections import namedtuple
from numba import void, int32, float32, float64
from numba import guvectorize
from numba import cuda
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
import warnings
from numba.core.errors import NumbaPerformanceWarning
from numba.tests.support import override_config
@guvectorize([void(float32[:], float32[:], float32[:])], '(x)->(x),(x)', target='cuda')
def copy_and_double(A, B, C):
    for i in range(B.size):
        B[i] = A[i]
        C[i] = A[i] * 2
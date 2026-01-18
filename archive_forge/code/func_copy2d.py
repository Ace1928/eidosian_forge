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
@guvectorize([void(float32[:, :], float32[:, :])], '(x, y)->(x, y)', target='cuda')
def copy2d(A, B):
    for x in range(B.shape[0]):
        for y in range(B.shape[1]):
            B[x, y] = A[x, y]
import numpy as np
from numba.core.utils import PYVERSION
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.tests.support import (override_config, captured_stderr,
from numba import cuda, float64
import unittest
def simple_cuda(A, B):
    i = cuda.grid(1)
    B[i] = A[i] + 1.5
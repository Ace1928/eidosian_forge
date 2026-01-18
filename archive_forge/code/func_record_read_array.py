import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def record_read_array(r, a):
    a[0] = r.h[0]
    a[1] = r.h[1]
import numpy as np
from numba import cuda
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
import unittest
from numba.np import numpy_support
def set_record(ary, i, j):
    ary[i] = ary[j]
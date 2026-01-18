import math
import itertools
import numpy as np
from numba.cuda.testing import unittest, CUDATestCase
from numba.core import types
from numba import cuda
from numba.tests.complex_usecases import (real_usecase, imag_usecase,
from numba.np import numpy_support
def basic_values(self):
    reals = [-0.0, +0.0, 1, -1, +1.5, -3.5, float('-inf'), float('+inf'), float('nan')]
    return [complex(x, y) for x, y in itertools.product(reals, reals)]
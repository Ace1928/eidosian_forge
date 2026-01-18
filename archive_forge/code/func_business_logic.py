import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
import numpy as np
@numba.jit
def business_logic(x, y, z):
    return 4 * z * (2 * x - 4 * y / 2 * pi)
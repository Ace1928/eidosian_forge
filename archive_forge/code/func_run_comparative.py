from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def run_comparative(compare_func, test_array):
    cfunc = njit(compare_func)
    numpy_result = compare_func(test_array)
    numba_result = cfunc(test_array)
    return (numpy_result, numba_result)
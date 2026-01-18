import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def test_global_complex_arr_npm(self):
    self.check_global_complex_arr(nopython=True)
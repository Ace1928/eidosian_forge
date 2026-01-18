import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def test_global_rec_arr_extract(self):
    self.check_global_rec_arr_extract(forceobj=True)
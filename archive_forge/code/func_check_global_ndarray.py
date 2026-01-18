import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def check_global_ndarray(self, **jitargs):
    ctestfunc = jit(**jitargs)(global_ndarray_func)
    self.assertEqual(ctestfunc(1), 11)
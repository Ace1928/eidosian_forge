import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def test_global_mixed_tuple(self):
    pyfunc = global_mixed_tuple
    jitfunc = njit(pyfunc)
    self.assertEqual(pyfunc(), jitfunc())
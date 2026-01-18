import numpy as np
from numba import jit, njit, errors
from numba.extending import register_jitable
from numba.tests import usecases
import unittest
def test_global_module(self):
    res = global_module_func(5, 6)
    self.assertEqual(True, res)
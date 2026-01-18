import ctypes
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import numpy as np, numpy_available
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface
def test_get_cntl(self):
    ma57 = MA57Interface()
    self.assertEqual(ma57.get_icntl(1), 6)
    self.assertEqual(ma57.get_icntl(7), 1)
    self.assertAlmostEqual(ma57.get_cntl(1), 0.01)
    self.assertAlmostEqual(ma57.get_cntl(2), 1e-20)
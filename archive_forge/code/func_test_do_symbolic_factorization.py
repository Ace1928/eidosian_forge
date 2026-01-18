import ctypes
import pyomo.common.unittest as unittest
from pyomo.contrib.pynumero.dependencies import numpy as np, numpy_available
from pyomo.contrib.pynumero.linalg.ma57 import MA57Interface
def test_do_symbolic_factorization(self):
    ma57 = MA57Interface()
    n = 5
    ne = 7
    irn = np.array([1, 1, 2, 2, 3, 3, 5], dtype=np.intc)
    jcn = np.array([1, 2, 3, 5, 3, 4, 5], dtype=np.intc)
    irn = irn - 1
    jcn = jcn - 1
    bad_jcn = np.array([1, 2, 3, 5, 3, 4], dtype=np.intc)
    ma57.do_symbolic_factorization(n, irn, jcn)
    self.assertEqual(ma57.get_info(1), 0)
    self.assertEqual(ma57.get_info(4), 0)
    self.assertEqual(ma57.get_info(9), 48)
    self.assertEqual(ma57.get_info(10), 53)
    self.assertEqual(ma57.get_info(14), 0)
    with self.assertRaisesRegex(AssertionError, 'Dimension mismatch'):
        ma57.do_symbolic_factorization(n, irn, bad_jcn)
import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
@skip_on_cudasim('Simulator does not track overloads')
def test_ambiguous_new_version(self):
    """Test compiling new version in an ambiguous case
        """
    c_add = cuda.jit(add_kernel)
    r = np.zeros(1, dtype=np.float64)
    INT = 1
    FLT = 1.5
    c_add[1, 1](r, INT, FLT)
    self.assertAlmostEqual(r[0], INT + FLT)
    self.assertEqual(len(c_add.overloads), 1)
    c_add[1, 1](r, FLT, INT)
    self.assertAlmostEqual(r[0], FLT + INT)
    self.assertEqual(len(c_add.overloads), 2)
    c_add[1, 1](r, FLT, FLT)
    self.assertAlmostEqual(r[0], FLT + FLT)
    self.assertEqual(len(c_add.overloads), 3)
    c_add[1, 1](r, 1, 1)
    self.assertAlmostEqual(r[0], INT + INT)
    self.assertEqual(len(c_add.overloads), 4, "didn't compile a new version")
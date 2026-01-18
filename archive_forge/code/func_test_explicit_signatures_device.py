import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_explicit_signatures_device(self):
    sigs = ['(int64, int64)', '(float64, float64)']
    f = self.add_device_usecase(sigs)
    r = np.zeros(1, dtype=np.int64)
    f[1, 1](r, 1, 2)
    self.assertPreciseEqual(r[0], 3)
    r = np.zeros(1, dtype=np.float64)
    f[1, 1](r, 1.5, 2.5)
    self.assertPreciseEqual(r[0], 4.0)
    if config.ENABLE_CUDASIM:
        return
    with self.assertRaises(TypingError) as cm:
        r = np.zeros(1, dtype=np.complex128)
        f[1, 1](r, 1j, 1j)
    msg = str(cm.exception)
    self.assertIn('Invalid use of type', msg)
    self.assertIn('with parameters (complex128, complex128)', msg)
    self.assertEqual(len(f.overloads), 2, f.overloads)
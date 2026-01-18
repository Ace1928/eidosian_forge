import numpy as np
from numba.cuda.testing import SerialMixin
from numba import typeof, cuda, njit
from numba.core.types import float64
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core import config
import unittest
@unittest.skipIf(not cuda.is_available(), 'NO CUDA')
@TestCase.run_test_in_subprocess(envvars={'NUMBA_BOUNDSCHECK': '1'})
def test_no_cuda_boundscheck(self):
    self.assertTrue(config.BOUNDSCHECK)
    with self.assertRaises(NotImplementedError):

        @cuda.jit(boundscheck=True)
        def func():
            pass

    @cuda.jit(boundscheck=False)
    def func3():
        pass

    @cuda.jit
    def func2(x, a):
        a[1] = x[1]
    a = np.ones((1,))
    x = np.zeros((1,))
    if not config.ENABLE_CUDASIM:
        func2[1, 1](x, a)
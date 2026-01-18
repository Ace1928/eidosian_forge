import unittest
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
from numba.tests.support import captured_stdout
def test_ex_cuda_ufunc_call(self):
    import numpy as np
    from numba import cuda

    @cuda.jit
    def f(r, x):
        np.sin(x, r)
    x = np.arange(10, dtype=np.float32) - 5
    r = np.zeros_like(x)
    f[1, 1](r, x)
    np.testing.assert_allclose(r, np.sin(x))
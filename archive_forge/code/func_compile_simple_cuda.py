import numpy as np
from numba.core.utils import PYVERSION
from numba.cuda.testing import skip_on_cudasim, CUDATestCase
from numba.tests.support import (override_config, captured_stderr,
from numba import cuda, float64
import unittest
def compile_simple_cuda(self):
    with captured_stderr() as err:
        with captured_stdout() as out:
            cfunc = cuda.jit((float64[:], float64[:]))(simple_cuda)
            A = np.linspace(0, 1, 10).astype(np.float64)
            B = np.zeros_like(A)
            cfunc[1, 10](A, B)
            self.assertTrue(np.allclose(A + 1.5, B))
    self.assertFalse(err.getvalue())
    return out.getvalue()
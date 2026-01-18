import numpy as np
from numba import cuda, int32, complex128, void
from numba.core import types
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from .extensions_usecases import test_struct_model_type, TestStruct
def test_local_array_1_tuple(self):
    """Ensure that local arrays can be constructed with 1-tuple shape
        """
    jculocal = cuda.jit('void(int32[:], int32[:])')(culocal1tuple)
    A = np.arange(5, dtype='int32')
    B = np.zeros_like(A)
    jculocal[1, 1](A, B)
    self.assertTrue(np.all(A == B))
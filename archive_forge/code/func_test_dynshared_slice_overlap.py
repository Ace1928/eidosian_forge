from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def test_dynshared_slice_overlap(self):

    @cuda.jit
    def slice_overlap(x):
        dynsmem = cuda.shared.array(0, dtype=int32)
        sm1 = dynsmem[0:2]
        sm2 = dynsmem[1:4]
        dynsmem[0] = 1
        dynsmem[1] = 2
        dynsmem[2] = 3
        dynsmem[3] = 4
        x[0] = sm1[0]
        x[1] = sm1[1]
        x[2] = sm2[0]
        x[3] = sm2[1]
        x[4] = sm2[2]
    arr = np.zeros(5, dtype=np.int32)
    expected = np.array([1, 2, 2, 3, 4], dtype=np.int32)
    self._test_dynshared_slice(slice_overlap, arr, expected)
from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def test_issue_953_sm_linkage_conflict(self):

    @cuda.jit(device=True)
    def inner():
        inner_arr = cuda.shared.array(1, dtype=int32)

    @cuda.jit
    def outer():
        outer_arr = cuda.shared.array(1, dtype=int32)
        inner()
    outer[1, 1]()
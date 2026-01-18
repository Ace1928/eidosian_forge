from numba import cuda, int32, float64, void
from numba.core.errors import TypingError
from numba.core import types
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
import numpy as np
from numba.np import numpy_support as nps
from .extensions_usecases import test_struct_model_type, TestStruct
def test_issue_fp16_support(self):
    self._check_shared_array_size_fp16(2, 2, types.float16)
    self._check_shared_array_size_fp16(2, 2, np.float16)
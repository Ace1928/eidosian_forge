import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@skip_on_cudasim('cudasim does not check signatures')
def test_declare_device_signature(self):
    f1 = cuda.declare_device('f1', int32(float32[:]))
    self._test_declare_device(f1)
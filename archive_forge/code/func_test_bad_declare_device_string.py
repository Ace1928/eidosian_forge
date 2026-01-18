import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@skip_on_cudasim('cudasim does not check signatures')
def test_bad_declare_device_string(self):
    with self.assertRaisesRegex(TypeError, 'Return type'):
        cuda.declare_device('f1', '(float32[:],)')
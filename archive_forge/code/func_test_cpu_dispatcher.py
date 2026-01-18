import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
def test_cpu_dispatcher(self):

    @jit
    def add(a, b):
        return a + b
    self._check_cpu_dispatcher(add)
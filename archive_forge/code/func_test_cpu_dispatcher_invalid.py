import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@skip_on_cudasim('not supported in cudasim')
def test_cpu_dispatcher_invalid(self):

    @jit('(i4, i4)')
    def add(a, b):
        return a + b
    with self.assertRaises(TypingError) as raises:
        self._check_cpu_dispatcher(add)
    msg = "Untyped global name 'add':.*using cpu function on device"
    expected = re.compile(msg)
    self.assertTrue(expected.search(str(raises.exception)) is not None)
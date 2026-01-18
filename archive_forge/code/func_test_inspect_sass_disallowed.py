import re
import types
import numpy as np
from numba.cuda.testing import unittest, skip_on_cudasim, CUDATestCase
from numba import cuda, jit, float32, int32
from numba.core.errors import TypingError
@skip_on_cudasim('not supported in cudasim')
def test_inspect_sass_disallowed(self):

    @cuda.jit(device=True)
    def foo(x, y):
        return x + y
    with self.assertRaises(RuntimeError) as raises:
        foo.inspect_sass((int32, int32))
    self.assertIn('Cannot inspect SASS of a device function', str(raises.exception))
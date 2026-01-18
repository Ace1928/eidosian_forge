from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import numpy as np
import unittest
@skip_on_cudasim('Recursion handled because simulator does not compile')
def test_growing_return_tuple(self):
    cfunc = self.mod.make_growing_tuple_case(cuda.jit)
    with self.assertRaises(TypingError) as raises:

        @cuda.jit('void()')
        def kernel():
            cfunc(100)
    self.assertIn('Return type of recursive function does not converge', str(raises.exception))
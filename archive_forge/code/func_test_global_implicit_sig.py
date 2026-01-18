from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import numpy as np
import unittest
def test_global_implicit_sig(self):
    self.check_fib(self.mod.fib3)
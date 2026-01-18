from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import CUDATestCase, skip_on_cudasim
import numpy as np
import unittest
def test_inner_explicit_sig(self):
    self.check_fib(self.mod.fib2)
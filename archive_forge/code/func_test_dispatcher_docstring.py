import numpy as np
import threading
from numba import boolean, config, cuda, float32, float64, int32, int64, void
from numba.core.errors import TypingError
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
import math
def test_dispatcher_docstring(self):

    @cuda.jit
    def add_kernel(a, b):
        """Add two integers, kernel version"""

    @cuda.jit(device=True)
    def add_device(a, b):
        """Add two integers, device version"""
    self.assertEqual('Add two integers, kernel version', add_kernel.__doc__)
    self.assertEqual('Add two integers, device version', add_device.__doc__)
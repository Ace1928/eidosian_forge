from numba import cuda
import numpy as np
from numba.cuda.testing import CUDATestCase
from numba.tests.support import override_config
import unittest
def test_device_jit_2(self):

    @cuda.jit(device=True)
    def inner(arg):
        return arg + 1

    @cuda.jit
    def outer(argin, argout):
        argout[0] = inner(argin[0]) + inner(2)
    a = np.zeros(1)
    b = np.zeros(1)
    stream = cuda.stream()
    d_a = cuda.to_device(a, stream)
    d_b = cuda.to_device(b, stream)
    outer[1, 1, stream](d_a, d_b)
    d_b.copy_to_host(b, stream)
    self.assertEqual(b[0], a[0] + 1 + (2 + 1))
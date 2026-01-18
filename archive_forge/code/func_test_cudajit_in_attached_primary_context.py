import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver
def test_cudajit_in_attached_primary_context(self):

    def do():
        from numba import cuda

        @cuda.jit
        def foo(a):
            for i in range(a.size):
                a[i] = i
        a = cuda.device_array(10)
        foo[1, 1](a)
        self.assertEqual(list(a.copy_to_host()), list(range(10)))
    self.test_attached_primary(do)
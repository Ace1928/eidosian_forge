import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver
def test_context_memory(self):
    try:
        mem = cuda.current_context().get_memory_info()
    except NotImplementedError:
        self.skipTest('EMM Plugin does not implement get_memory_info()')
    self.assertIsInstance(mem.free, numbers.Number)
    self.assertEqual(mem.free, mem[0])
    self.assertIsInstance(mem.total, numbers.Number)
    self.assertEqual(mem.total, mem[1])
    self.assertLessEqual(mem.free, mem.total)
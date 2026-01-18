import numbers
from ctypes import byref
import weakref
from numba import cuda
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.cuda.cudadrv import driver
def test_gpus_current(self):
    self.assertIs(cuda.gpus.current, None)
    with cuda.gpus[0]:
        self.assertEqual(int(cuda.gpus.current.id), 0)
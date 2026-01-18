import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
def test_get_memory_info(self):
    ctx = cuda.current_context()
    meminfo = ctx.get_memory_info()
    self.assertTrue(ctx.memory_manager.get_memory_info_called)
    self.assertEqual(meminfo.free, 32)
    self.assertEqual(meminfo.total, 64)
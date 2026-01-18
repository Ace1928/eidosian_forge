import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
def test_bad_plugin_version(self):
    with self.assertRaises(RuntimeError) as raises:
        cuda.set_memory_manager(BadVersionEMMPlugin)
    self.assertIn('version 1 required', str(raises.exception))
import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
@linux_only
def test_get_ipc_handle(self):
    arr = np.arange(2)
    d_arr = cuda.device_array_like(arr)
    ipch = d_arr.get_ipc_handle()
    ctx = cuda.current_context()
    self.assertTrue(ctx.memory_manager.get_ipc_handle_called)
    self.assertIn('Dummy IPC handle for alloc 1', ipch._ipc_handle)
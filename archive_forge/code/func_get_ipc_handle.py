import ctypes
import numpy as np
import weakref
from numba import cuda
from numba.core import config
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
from numba.tests.support import linux_only
def get_ipc_handle(self, memory):
    self.get_ipc_handle_called = True
    return 'Dummy IPC handle for alloc %s' % memory.device_pointer.value
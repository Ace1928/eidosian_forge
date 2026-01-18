import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def list_iter(self, itptr):
    self.tc.numba_list_iter(itptr, self.lp)
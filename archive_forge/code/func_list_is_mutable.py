import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def list_is_mutable(self):
    return self.tc.numba_list_is_mutable(self.lp)
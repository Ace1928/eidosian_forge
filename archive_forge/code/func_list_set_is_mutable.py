import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def list_set_is_mutable(self, is_mutable):
    return self.tc.numba_list_set_is_mutable(self.lp, is_mutable)
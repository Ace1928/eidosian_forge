import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def set_mutable(self):
    return self.list_set_is_mutable(1)
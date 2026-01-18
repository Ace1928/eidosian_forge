import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
@property
def is_mutable(self):
    return self.list_is_mutable()
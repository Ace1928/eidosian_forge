import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def make_item(v):
    tmp = '{:0{}}'.format(nmax - v - 1, item_size).encode('latin-1')
    return tmp[:item_size]
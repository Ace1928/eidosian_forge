import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
def deg(self):
    """return the degree of the polynomial"""
    a = self.p
    if a == 0:
        return -1
    n = 0
    while a >= 65536:
        n += 16
        a = a >> 16
    a = int(a)
    while a > 1:
        n += 1
        a = a >> 1
    return n
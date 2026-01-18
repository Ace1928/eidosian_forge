import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
def crc8p(d):
    p = 0
    for i in d:
        p = p * 256 + i
    p = poly(p)
    return int(p * x8p % g8p)
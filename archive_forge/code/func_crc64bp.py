import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
def crc64bp(d):
    p = 0
    for i in d:
        p = p * 256 + i
    p = poly(p)
    return int(p * x64p % g64bp)
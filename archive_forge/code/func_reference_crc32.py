import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
@staticmethod
def reference_crc32(d, crc=0):
    """This function modifies the return value of binascii.crc32
        to be an unsigned 32-bit value. I.e. in the range 0 to 2**32-1."""
    if crc > 2147483647:
        x = int(crc & 2147483647)
        crc = x | -2147483648
    x = binascii.crc32(d, crc)
    return int(x) & 4294967295
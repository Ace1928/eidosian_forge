import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
def test_bytearray_input(self):
    """Test that bytearray inputs are accepted, as an example
        of a type that implements the buffer protocol."""
    for crc_name in self.check_crc_names:
        crcfun = mkPredefinedCrcFun(crc_name)
        for i in range(len(self.msg) + 1):
            test_msg = self.msg[:i]
            bytes_answer = crcfun(test_msg)
            bytearray_answer = crcfun(bytearray(test_msg))
            self.assertEqual(bytes_answer, bytearray_answer)
import unittest
from array import array
import binascii
from .crcmod import mkCrcFun, Crc
from .crcmod import _usingExtension
from .predefined import PredefinedCrc
from .predefined import mkPredefinedCrcFun
from .predefined import _crc_definitions as _predefined_crc_definitions
def test_known_answers(self):
    for crcfun_name, v in self.known_answers:
        crcfun = mkPredefinedCrcFun(crcfun_name)
        self.assertEqual(crcfun(b'', 0), 0, "Wrong answer for CRC '%s', input ''" % crcfun_name)
        for i, msg in enumerate(self.test_messages_for_known_answers):
            self.assertEqual(crcfun(msg), v[i], "Wrong answer for CRC %s, input '%s'" % (crcfun_name, msg))
            self.assertEqual(crcfun(msg[4:], crcfun(msg[:4])), v[i], "Wrong answer for CRC %s, input '%s'" % (crcfun_name, msg))
            self.assertEqual(crcfun(msg[-1:], crcfun(msg[:-1])), v[i], "Wrong answer for CRC %s, input '%s'" % (crcfun_name, msg))
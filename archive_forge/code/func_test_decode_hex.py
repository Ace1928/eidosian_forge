import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_hex(self):
    self.assertEqual(_b(''), self.decoder.write(_b('A\r\n12345678900\r\n')))
    self.assertEqual(_b('1234567890'), self.output.getvalue())
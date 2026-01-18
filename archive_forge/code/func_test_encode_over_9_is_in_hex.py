import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_encode_over_9_is_in_hex(self):
    self.encoder.write(_b('1234567890'))
    self.encoder.close()
    self.assertEqual(_b('A\r\n12345678900\r\n'), self.output.getvalue())
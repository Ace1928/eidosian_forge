import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_nothing(self):
    self.assertEqual(_b(''), self.decoder.write(_b('0\r\n')))
    self.assertEqual(_b(''), self.output.getvalue())
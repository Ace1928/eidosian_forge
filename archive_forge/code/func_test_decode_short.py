import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_short(self):
    self.assertEqual(_b(''), self.decoder.write(_b('3\r\nabc0\r\n')))
    self.assertEqual(_b('abc'), self.output.getvalue())
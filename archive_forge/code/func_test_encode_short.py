import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_encode_short(self):
    self.encoder.write(_b('abc'))
    self.encoder.close()
    self.assertEqual(_b('3\r\nabc0\r\n'), self.output.getvalue())
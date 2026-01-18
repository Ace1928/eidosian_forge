import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_decode_excess_bytes_from_write(self):
    self.assertEqual(_b('1234'), self.decoder.write(_b('3\r\nabc0\r\n1234')))
    self.assertEqual(_b('abc'), self.output.getvalue())
import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_encode_nothing(self):
    self.encoder.close()
    self.assertEqual(_b('0\r\n'), self.output.getvalue())
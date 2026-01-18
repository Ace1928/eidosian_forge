import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_close_body_short_errors(self):
    self.assertEqual(None, self.decoder.write(_b('2\r\na')))
    self.assertRaises(ValueError, self.decoder.close)
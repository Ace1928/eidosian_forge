import unittest
from io import BytesIO
from testtools.compat import _b
import subunit.chunked
def test_close_after_finished_stream_safe(self):
    self.assertEqual(None, self.decoder.write(_b('2\r\nab')))
    self.assertEqual(_b(''), self.decoder.write(_b('0\r\n')))
    self.decoder.close()
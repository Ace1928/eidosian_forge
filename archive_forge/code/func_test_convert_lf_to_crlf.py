from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_convert_lf_to_crlf(self):
    self.assertEqual(convert_lf_to_crlf(b'line1\nline2'), b'line1\r\nline2')
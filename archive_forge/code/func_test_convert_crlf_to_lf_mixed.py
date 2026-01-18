from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_convert_crlf_to_lf_mixed(self):
    self.assertEqual(convert_crlf_to_lf(b'line1\r\n\nline2'), b'line1\n\nline2')
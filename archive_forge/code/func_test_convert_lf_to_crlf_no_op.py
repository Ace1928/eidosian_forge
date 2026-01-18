from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_convert_lf_to_crlf_no_op(self):
    self.assertEqual(convert_lf_to_crlf(b'foobar'), b'foobar')
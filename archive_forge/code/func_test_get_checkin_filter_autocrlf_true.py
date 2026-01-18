from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_get_checkin_filter_autocrlf_true(self):
    checkin_filter = get_checkin_filter_autocrlf(b'true')
    self.assertEqual(checkin_filter, convert_crlf_to_lf)
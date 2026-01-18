from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_get_checkout_filter_autocrlf_default(self):
    checkout_filter = get_checkout_filter_autocrlf(b'false')
    self.assertEqual(checkout_filter, None)
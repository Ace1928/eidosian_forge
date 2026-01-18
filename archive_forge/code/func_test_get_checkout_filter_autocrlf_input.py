from dulwich.tests import TestCase
from ..line_ending import (
from ..objects import Blob
def test_get_checkout_filter_autocrlf_input(self):
    checkout_filter = get_checkout_filter_autocrlf(b'input')
    self.assertEqual(checkout_filter, None)
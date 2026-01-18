import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_decode_not_start(self):
    self.assertDecode(1, 1, 6, b'abc\x91\x01\x01def', 3)
    self.assertDecode(9, 10, 5, b'ab\x91\t\nde', 2)
    self.assertDecode(254, 255, 6, b'not\x91\xfe\xffcopy', 3)
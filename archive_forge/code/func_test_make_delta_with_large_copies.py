import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_make_delta_with_large_copies(self):
    big_text = _text3 * 1220
    delta = self.make_delta(big_text, big_text)
    self.assertDeltaIn(b'\xdc\x86\n\x80\x84\x01\xb4\x02\\\x83', None, delta)
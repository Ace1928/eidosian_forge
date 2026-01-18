import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_apply_delta(self):
    target = self.apply_delta(_text1, b'N\x90/\x1fdiffer from\nagainst other text\n')
    self.assertEqual(_text2, target)
    target = self.apply_delta(_text2, b'M\x90/\x1ebe matched\nagainst other text\n')
    self.assertEqual(_text1, target)
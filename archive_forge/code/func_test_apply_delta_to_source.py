import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_apply_delta_to_source(self):
    source_and_delta = _text1 + b'N\x90/\x1fdiffer from\nagainst other text\n'
    self.assertEqual(_text2, self.apply_delta_to_source(source_and_delta, len(_text1), len(source_and_delta)))
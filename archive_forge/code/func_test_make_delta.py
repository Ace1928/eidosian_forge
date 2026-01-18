import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_make_delta(self):
    di = self._gc_module.DeltaIndex(_text1)
    delta = di.make_delta(_text2)
    self.assertEqual(b'N\x90/\x1fdiffer from\nagainst other text\n', delta)
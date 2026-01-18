import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test_first_add_source_doesnt_index_until_make_delta(self):
    di = self._gc_module.DeltaIndex()
    self.assertFalse(di._has_index())
    di.add_source(_text1, 0)
    self.assertFalse(di._has_index())
    delta = di.make_delta(_text2)
    self.assertTrue(di._has_index())
    self.assertEqual(b'N\x90/\x1fdiffer from\nagainst other text\n', delta)
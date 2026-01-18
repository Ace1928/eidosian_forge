import sys
from ... import tests
from ...tests import features
from ...tests.scenarios import load_tests_apply_scenarios
from .. import _groupcompress_py
def test__dump_no_index(self):
    di = self._gc_module.DeltaIndex()
    self.assertEqual(None, di._dump_index())
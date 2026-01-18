import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_complex_structure_missing(self):
    state = self.create_complex_dirstate()
    self.addCleanup(state.unlock)
    self.assertEntryEqual(None, None, None, state, b'_', 0)
    self.assertEntryEqual(None, None, None, state, b'_\xc3\xa5', 0)
    self.assertEntryEqual(None, None, None, state, b'a/b', 0)
    self.assertEntryEqual(None, None, None, state, b'c/d', 0)
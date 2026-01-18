import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_iter_children_b(self):
    state, dirblocks = self.create_dirstate_with_two_trees()
    self.addCleanup(state.unlock)
    expected_result = []
    expected_result.append(dirblocks[3][1][2])
    expected_result.append(dirblocks[3][1][3])
    expected_result.append(dirblocks[3][1][4])
    self.assertEqual(expected_result, list(state._iter_child_entries(1, b'b')))
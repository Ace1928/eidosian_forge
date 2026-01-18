import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_construct_with_path(self):
    tree = self.make_branch_and_tree('tree')
    state = dirstate.DirState.from_tree(tree, 'dirstate.from_tree')
    lines = state.get_lines()
    state.unlock()
    self.build_tree_contents([('dirstate', b''.join(lines))])
    expected_result = ([], [((b'', b'', tree.path2id('')), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)])])
    state = dirstate.DirState.on_file('dirstate')
    state.lock_write()
    self.check_state_with_reopen(expected_result, state)
import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_non_empty_no_parents_to_dirstate(self):
    """We should be able to create a dirstate for an empty tree."""
    tree = self.get_tree_with_a_file()
    expected_result = ([], [((b'', b'', tree.path2id('')), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT)]), ((b'', b'a file', b'a-file-id'), [(b'f', b'', 0, False, dirstate.DirState.NULLSTAT)])])
    state = dirstate.DirState.from_tree(tree, 'dirstate')
    self.check_state_with_reopen(expected_result, state)
import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_2_parents_not_empty_to_dirstate(self):
    tree = self.get_tree_with_a_file()
    rev_id = tree.commit('first post')
    tree2 = tree.controldir.sprout('tree2').open_workingtree()
    self.build_tree_contents([('tree2/a file', b'merge content\n')])
    rev_id2 = tree2.commit('second post')
    tree.merge_from_branch(tree2.branch)
    self.build_tree_contents([('tree/a file', b'new content\n')])
    expected_result = ([rev_id, rev_id2], [((b'', b'', tree.path2id('')), [(b'd', b'', 0, False, dirstate.DirState.NULLSTAT), (b'd', b'', 0, False, rev_id), (b'd', b'', 0, False, rev_id)]), ((b'', b'a file', b'a-file-id'), [(b'f', b'', 0, False, dirstate.DirState.NULLSTAT), (b'f', b'c3ed76e4bfd45ff1763ca206055bca8e9fc28aa8', 24, False, rev_id), (b'f', b'314d796174c9412647c3ce07dfb5d36a94e72958', 14, False, rev_id2)])])
    state = dirstate.DirState.from_tree(tree, 'dirstate')
    self.check_state_with_reopen(expected_result, state)
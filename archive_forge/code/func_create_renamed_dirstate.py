import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_renamed_dirstate(self):
    """Create a dirstate with a few internal renames.

        This takes the basic dirstate, and moves the paths around.
        """
    tree, state, expected = self.create_basic_dirstate()
    tree.rename_one('a', 'b/g')
    tree.rename_one('b/d', 'h')
    old_a = expected[b'a']
    expected[b'a'] = (old_a[0], [(b'r', b'b/g', 0, False, b''), old_a[1][1]])
    expected[b'b/g'] = ((b'b', b'g', b'a-id'), [old_a[1][0], (b'r', b'a', 0, False, b'')])
    old_d = expected[b'b/d']
    expected[b'b/d'] = (old_d[0], [(b'r', b'h', 0, False, b''), old_d[1][1]])
    expected[b'h'] = ((b'', b'h', b'd-id'), [old_d[1][0], (b'r', b'b/d', 0, False, b'')])
    old_e = expected[b'b/d/e']
    expected[b'b/d/e'] = (old_e[0], [(b'r', b'h/e', 0, False, b''), old_e[1][1]])
    expected[b'h/e'] = ((b'h', b'e', b'e-id'), [old_e[1][0], (b'r', b'b/d/e', 0, False, b'')])
    state.unlock()
    try:
        new_state = dirstate.DirState.from_tree(tree, 'dirstate')
        try:
            new_state.save()
        finally:
            new_state.unlock()
    finally:
        state.lock_read()
    return (tree, state, expected)
import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def create_basic_dirstate(self):
    """Create a dirstate with a few files and directories.

            a
            b/
              c
              d/
                e
            b-c
            f
        """
    tree = self.make_branch_and_tree('tree')
    paths = ['a', 'b/', 'b/c', 'b/d/', 'b/d/e', 'b-c', 'f']
    file_ids = [b'a-id', b'b-id', b'c-id', b'd-id', b'e-id', b'b-c-id', b'f-id']
    self.build_tree(['tree/' + p for p in paths])
    tree.set_root_id(b'TREE_ROOT')
    tree.add([p.rstrip('/') for p in paths], ids=file_ids)
    tree.commit('initial', rev_id=b'rev-1')
    revision_id = b'rev-1'
    t = self.get_transport('tree')
    a_text = t.get_bytes('a')
    a_sha = osutils.sha_string(a_text)
    a_len = len(a_text)
    c_text = t.get_bytes('b/c')
    c_sha = osutils.sha_string(c_text)
    c_len = len(c_text)
    e_text = t.get_bytes('b/d/e')
    e_sha = osutils.sha_string(e_text)
    e_len = len(e_text)
    b_c_text = t.get_bytes('b-c')
    b_c_sha = osutils.sha_string(b_c_text)
    b_c_len = len(b_c_text)
    f_text = t.get_bytes('f')
    f_sha = osutils.sha_string(f_text)
    f_len = len(f_text)
    null_stat = dirstate.DirState.NULLSTAT
    expected = {b'': ((b'', b'', b'TREE_ROOT'), [(b'd', b'', 0, False, null_stat), (b'd', b'', 0, False, revision_id)]), b'a': ((b'', b'a', b'a-id'), [(b'f', b'', 0, False, null_stat), (b'f', a_sha, a_len, False, revision_id)]), b'b': ((b'', b'b', b'b-id'), [(b'd', b'', 0, False, null_stat), (b'd', b'', 0, False, revision_id)]), b'b/c': ((b'b', b'c', b'c-id'), [(b'f', b'', 0, False, null_stat), (b'f', c_sha, c_len, False, revision_id)]), b'b/d': ((b'b', b'd', b'd-id'), [(b'd', b'', 0, False, null_stat), (b'd', b'', 0, False, revision_id)]), b'b/d/e': ((b'b/d', b'e', b'e-id'), [(b'f', b'', 0, False, null_stat), (b'f', e_sha, e_len, False, revision_id)]), b'b-c': ((b'', b'b-c', b'b-c-id'), [(b'f', b'', 0, False, null_stat), (b'f', b_c_sha, b_c_len, False, revision_id)]), b'f': ((b'', b'f', b'f-id'), [(b'f', b'', 0, False, null_stat), (b'f', f_sha, f_len, False, revision_id)])}
    state = dirstate.DirState.from_tree(tree, 'dirstate')
    try:
        state.save()
    finally:
        state.unlock()
    state = dirstate.DirState.on_file('dirstate')
    state.lock_read()
    self.addCleanup(state.unlock)
    self.assertEqual(dirstate.DirState.NOT_IN_MEMORY, state._dirblock_state)
    state._bisect_page_size = 200
    return (tree, state, expected)
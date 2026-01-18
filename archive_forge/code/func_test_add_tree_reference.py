import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_add_tree_reference(self):
    state = dirstate.DirState.initialize('dirstate')
    expected_entry = ((b'', b'subdir', b'subdir-id'), [(b't', b'subtree-123123', 0, False, b'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')])
    try:
        state.add('subdir', b'subdir-id', 'tree-reference', None, b'subtree-123123')
        entry = state._get_entry(0, b'subdir-id', b'subdir')
        self.assertEqual(entry, expected_entry)
        state._validate()
        state.save()
    finally:
        state.unlock()
    state.lock_read()
    self.addCleanup(state.unlock)
    state._validate()
    entry2 = state._get_entry(0, b'subdir-id', b'subdir')
    self.assertEqual(entry, entry2)
    self.assertEqual(entry, expected_entry)
    entry2 = state._get_entry(0, fileid_utf8=b'subdir-id')
    self.assertEqual(entry, expected_entry)
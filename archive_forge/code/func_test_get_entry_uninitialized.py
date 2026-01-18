import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_get_entry_uninitialized(self):
    """Calling get_entry will load data if it needs to"""
    state = self.create_dirstate_with_root()
    try:
        state.save()
    finally:
        state.unlock()
    del state
    state = dirstate.DirState.on_file('dirstate')
    state.lock_read()
    try:
        self.assertEqual(dirstate.DirState.NOT_IN_MEMORY, state._header_state)
        self.assertEqual(dirstate.DirState.NOT_IN_MEMORY, state._dirblock_state)
        self.assertEntryEqual(b'', b'', b'a-root-value', state, b'', 0)
    finally:
        state.unlock()
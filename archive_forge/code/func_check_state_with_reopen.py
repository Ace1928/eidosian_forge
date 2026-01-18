import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def check_state_with_reopen(self, expected_result, state):
    """Check that state has current state expected_result.

        This will check the current state, open the file anew and check it
        again.
        This function expects the current state to be locked for writing, and
        will unlock it before re-opening.
        This is required because we can't open a lock_read() while something
        else has a lock_write().
            write => mutually exclusive lock
            read => shared lock
        """
    self.assertTrue(state._lock_token is not None)
    try:
        self.assertEqual(expected_result[0], state.get_parent_ids())
        self.assertEqual([], state.get_ghosts())
        self.assertEqual(expected_result[1], list(state._iter_entries()))
        state.save()
    finally:
        state.unlock()
    del state
    state = dirstate.DirState.on_file('dirstate')
    state.lock_read()
    try:
        self.assertEqual(expected_result[1], list(state._iter_entries()))
    finally:
        state.unlock()
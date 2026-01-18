import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def assertBadDelta(self, active, basis, delta):
    """Test that we raise InconsistentDelta when appropriate.

        :param active: The active tree shape
        :param basis: The basis tree shape
        :param delta: A description of the delta to apply. Similar to the form
            for regular inventory deltas, but omitting the InventoryEntry.
            So adding a file is: (None, 'path', b'file-id')
            Adding a directory is: (None, 'path/', b'dir-id')
            Renaming a dir is: ('old/', 'new/', b'dir-id')
            etc.
        """
    active_tree = self.create_tree_from_shape(b'active', active)
    basis_tree = self.create_tree_from_shape(b'basis', basis)
    inv_delta = self.create_inv_delta(delta, b'target')
    state = self.create_empty_dirstate()
    state.set_state_from_scratch(active_tree.root_inventory, [(b'basis', basis_tree)], [])
    self.assertRaises(errors.InconsistentDelta, state.update_basis_by_delta, inv_delta, b'target')
    self.assertTrue(state._changes_aborted)
import os
import tempfile
from ... import controldir, errors, memorytree, osutils
from ... import revision as _mod_revision
from ... import revisiontree, tests
from ...tests import features, test_osutils
from ...tests.scenarios import load_tests_apply_scenarios
from .. import dirstate, inventory, inventorytree, workingtree_4
def test_add_path_to_unversioned_directory(self):
    """Adding a path to an unversioned directory should error.

        This is a duplicate of TestWorkingTree.test_add_in_unversioned,
        once dirstate is stable and if it is merged with WorkingTree3, consider
        removing this copy of the test.
        """
    self.build_tree(['unversioned/', 'unversioned/a file'])
    state = dirstate.DirState.initialize('dirstate')
    self.addCleanup(state.unlock)
    self.assertRaises(errors.NotVersionedError, state.add, 'unversioned/a file', b'a-file-id', 'file', None, None)
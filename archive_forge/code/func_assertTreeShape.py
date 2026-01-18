from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def assertTreeShape(self, expected_shape, tree):
    """Check that the tree shape matches expectations."""
    tree.lock_read()
    try:
        entries = [(path, ie.file_id, ie.kind) for path, ie in tree.iter_entries_by_dir()]
    finally:
        tree.unlock()
    self.assertEqual(expected_shape, entries)
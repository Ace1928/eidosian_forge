from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_get_branch(self):
    """get_branch returns the created branch."""
    builder = BranchBuilder(self.get_transport().clone('foo'))
    branch = builder.get_branch()
    self.assertIsInstance(branch, _mod_branch.Branch)
    self.assertEqual(self.get_transport().clone('foo').base, branch.base)
    self.assertEqual((0, _mod_revision.NULL_REVISION), branch.last_revision_info())
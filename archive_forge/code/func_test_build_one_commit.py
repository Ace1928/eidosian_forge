from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_build_one_commit(self):
    """doing build_commit causes a commit to happen."""
    builder = BranchBuilder(self.get_transport().clone('foo'))
    rev_id = builder.build_commit()
    branch = builder.get_branch()
    self.assertEqual((1, rev_id), branch.last_revision_info())
    self.assertEqual('commit 1', branch.repository.get_revision(branch.last_revision()).message)
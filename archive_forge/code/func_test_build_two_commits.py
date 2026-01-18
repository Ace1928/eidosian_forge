from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_build_two_commits(self):
    """The second commit has the right parents and message."""
    builder = BranchBuilder(self.get_transport().clone('foo'))
    rev_id1 = builder.build_commit()
    rev_id2 = builder.build_commit()
    branch = builder.get_branch()
    self.assertEqual((2, rev_id2), branch.last_revision_info())
    self.assertEqual('commit 2', branch.repository.get_revision(branch.last_revision()).message)
    self.assertEqual([rev_id1], branch.repository.get_revision(branch.last_revision()).parent_ids)
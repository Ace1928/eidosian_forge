from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_build_commit_parent_ids(self):
    """build_commit() takes a parent_ids argument."""
    builder = BranchBuilder(self.get_transport().clone('foo'))
    rev_id1 = builder.build_commit(parent_ids=[b'ghost'], allow_leftmost_as_ghost=True)
    rev_id2 = builder.build_commit(parent_ids=[])
    branch = builder.get_branch()
    self.assertEqual((1, rev_id2), branch.last_revision_info())
    self.assertEqual([b'ghost'], branch.repository.get_revision(rev_id1).parent_ids)
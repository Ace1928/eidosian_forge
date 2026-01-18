from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_commit_timestamp(self):
    builder = self.make_branch_builder('foo')
    rev_id = builder.build_snapshot(None, [('add', ('', None, 'directory', None))], timestamp=1234567890)
    rev = builder.get_branch().repository.get_revision(rev_id)
    self.assertEqual(1234567890, int(rev.timestamp))
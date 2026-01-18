from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_commit_message_supplied(self):
    builder = BranchBuilder(self.get_transport().clone('foo'))
    rev_id = builder.build_snapshot(None, [('add', ('', None, 'directory', None))], message='Foo')
    branch = builder.get_branch()
    rev = branch.repository.get_revision(rev_id)
    self.assertEqual('Foo', rev.message)
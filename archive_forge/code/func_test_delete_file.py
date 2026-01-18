from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_delete_file(self):
    builder = self.build_a_rev()
    rev_id2 = builder.build_snapshot(None, [('unversion', 'a')], revision_id=b'B-id')
    self.assertEqual(b'B-id', rev_id2)
    branch = builder.get_branch()
    rev_tree = branch.repository.revision_tree(rev_id2)
    rev_tree.lock_read()
    self.addCleanup(rev_tree.unlock)
    self.assertTreeShape([('', b'a-root-id', 'directory')], rev_tree)
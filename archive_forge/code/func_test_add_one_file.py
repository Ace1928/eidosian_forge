from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_add_one_file(self):
    builder = self.build_a_rev()
    branch = builder.get_branch()
    self.assertEqual((1, b'A-id'), branch.last_revision_info())
    rev_tree = branch.repository.revision_tree(b'A-id')
    rev_tree.lock_read()
    self.addCleanup(rev_tree.unlock)
    self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file')], rev_tree)
    self.assertEqual(b'contents', rev_tree.get_file_text('a'))
from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_delete_directory(self):
    builder = self.build_a_rev()
    rev_id2 = builder.build_snapshot(None, [('add', ('b', b'b-id', 'directory', None)), ('add', ('b/c', b'c-id', 'file', b'foo\n')), ('add', ('b/d', b'd-id', 'directory', None)), ('add', ('b/d/e', b'e-id', 'file', b'eff\n'))], revision_id=b'B-id')
    rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
    self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('b', b'b-id', 'directory'), ('b/c', b'c-id', 'file'), ('b/d', b'd-id', 'directory'), ('b/d/e', b'e-id', 'file')], rev_tree)
    builder.build_snapshot(None, [('unversion', 'b')], revision_id=b'C-id')
    rev_tree = builder.get_branch().repository.revision_tree(b'C-id')
    self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file')], rev_tree)
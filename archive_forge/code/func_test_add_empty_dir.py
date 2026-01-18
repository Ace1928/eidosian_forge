from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_add_empty_dir(self):
    builder = self.build_a_rev()
    rev_id2 = builder.build_snapshot(None, [('add', ('b', b'b-id', 'directory', None))], revision_id=b'B-id')
    rev_tree = builder.get_branch().repository.revision_tree(b'B-id')
    self.assertTreeShape([('', b'a-root-id', 'directory'), ('a', b'a-id', 'file'), ('b', b'b-id', 'directory')], rev_tree)
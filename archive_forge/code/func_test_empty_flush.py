from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def test_empty_flush(self):
    """A flush with no actions before it is a no-op."""
    builder = BranchBuilder(self.get_transport().clone('foo'))
    builder.start_series()
    builder.build_snapshot(None, [('add', ('', b'TREE_ROOT', 'directory', ''))], revision_id=b'rev-1')
    builder.build_snapshot(None, [('flush', None)], revision_id=b'rev-2')
    builder.finish_series()
    rev_tree = builder.get_branch().repository.revision_tree(b'rev-2')
    self.assertTreeShape([('', b'TREE_ROOT', 'directory')], rev_tree)
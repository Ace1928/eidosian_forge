from .. import branch as _mod_branch
from .. import revision as _mod_revision
from .. import tests
from ..branchbuilder import BranchBuilder
from ..bzr import branch as _mod_bzrbranch
def build_a_rev(self):
    builder = BranchBuilder(self.get_transport().clone('foo'))
    rev_id1 = builder.build_snapshot(None, [('add', ('', b'a-root-id', 'directory', None)), ('add', ('a', b'a-id', 'file', b'contents'))], revision_id=b'A-id')
    self.assertEqual(b'A-id', rev_id1)
    return builder
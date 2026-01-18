import os
from breezy import trace
from breezy.rename_map import RenameMap
from breezy.tests import TestCaseWithTransport
def test_guess_renames_preserves_children(self):
    """When a directory has been moved, its children are preserved."""
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree_contents([('tree/foo/', b''), ('tree/foo/bar', b'bar'), ('tree/foo/empty', b'')])
    tree.add(['foo', 'foo/bar', 'foo/empty'], ids=[b'foo-id', b'bar-id', b'empty-id'])
    tree.commit('rev1')
    os.rename('tree/foo', 'tree/baz')
    RenameMap.guess_renames(tree.basis_tree(), tree)
    self.assertEqual('baz/empty', tree.id2path(b'empty-id'))
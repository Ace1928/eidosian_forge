import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def prepare_shelve_deletion(self):
    tree = self.make_branch_and_tree('tree')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree_contents([('tree/foo/',), ('tree/foo/bar', b'baz')])
    tree.add(['foo', 'foo/bar'], ids=[b'foo-id', b'bar-id'])
    tree.commit('Added file and directory')
    tree.unversion(['foo', 'foo/bar'])
    os.unlink('tree/foo/bar')
    os.rmdir('tree/foo')
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('delete file', b'bar-id', 'file', 'foo/bar'), ('delete file', b'foo-id', 'directory', 'foo')], sorted(list(creator.iter_shelvable())))
    return (creator, tree)
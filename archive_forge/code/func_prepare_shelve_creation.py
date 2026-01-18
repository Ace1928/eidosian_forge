import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def prepare_shelve_creation(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.commit('Empty tree')
    self.build_tree_contents([('foo', b'a\n'), ('bar/',)])
    tree.add(['foo', 'bar'], ids=[b'foo-id', b'bar-id'])
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('add file', b'bar-id', 'directory', 'bar'), ('add file', b'foo-id', 'file', 'foo')], sorted(list(creator.iter_shelvable())))
    return (creator, tree)
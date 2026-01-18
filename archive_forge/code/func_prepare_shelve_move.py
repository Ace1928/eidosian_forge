import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def prepare_shelve_move(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo/', 'bar/', 'foo/baz'])
    tree.add(['foo', 'bar', 'foo/baz'], ids=[b'foo-id', b'bar-id', b'baz-id'])
    tree.commit('foo')
    tree.rename_one('foo/baz', 'bar/baz')
    tree.lock_tree_write()
    self.addCleanup(tree.unlock)
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    self.assertEqual([('rename', b'baz-id', 'foo/baz', 'bar/baz')], list(creator.iter_shelvable()))
    return (creator, tree)
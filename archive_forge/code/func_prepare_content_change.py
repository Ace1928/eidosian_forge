import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def prepare_content_change(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree_contents([('foo', b'a\n')])
    tree.add('foo', ids=b'foo-id')
    tree.commit('Committed foo')
    self.build_tree_contents([('foo', b'b\na\nc\n')])
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    return creator
import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_make_merger(self):
    tree = self.make_branch_and_tree('tree')
    tree.commit('first commit')
    self.build_tree_contents([('tree/foo', b'bar')])
    tree.lock_write()
    self.addCleanup(tree.unlock)
    tree.add('foo', ids=b'foo-id')
    creator = shelf.ShelfCreator(tree, tree.basis_tree())
    self.addCleanup(creator.finalize)
    list(creator.iter_shelvable())
    creator.shelve_creation(b'foo-id')
    with open('shelf-file', 'w+b') as shelf_file:
        creator.write_shelf(shelf_file)
        creator.transform()
        shelf_file.seek(0)
        unshelver = shelf.Unshelver.from_tree_and_shelf(tree, shelf_file)
        unshelver.make_merger().do_merge()
        self.addCleanup(unshelver.finalize)
        self.assertFileEqual(b'bar', 'tree/foo')
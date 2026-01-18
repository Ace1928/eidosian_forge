import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_corrupt_shelf(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('shelf', EMPTY_SHELF.replace(b'metadata', b'foo'))])
    shelf_file = open('shelf', 'rb')
    self.addCleanup(shelf_file.close)
    e = self.assertRaises(shelf.ShelfCorrupt, shelf.Unshelver.from_tree_and_shelf, tree, shelf_file)
    self.assertEqual('Shelf corrupt.', str(e))
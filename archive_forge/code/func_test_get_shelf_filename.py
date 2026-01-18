import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_get_shelf_filename(self):
    tree = self.make_branch_and_tree('.')
    manager = tree.get_shelf_manager()
    self.assertEqual('shelf-1', manager.get_shelf_filename(1))
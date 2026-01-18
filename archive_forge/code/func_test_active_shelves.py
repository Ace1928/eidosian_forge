import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_active_shelves(self):
    manager = self.get_manager()
    self.assertEqual([], manager.active_shelves())
    shelf_id, shelf_file = manager.new_shelf()
    shelf_file.close()
    self.assertEqual([1], manager.active_shelves())
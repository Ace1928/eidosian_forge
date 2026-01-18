import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_new_shelf(self):
    manager = self.get_manager()
    shelf_id, shelf_file = manager.new_shelf()
    shelf_file.close()
    self.assertEqual(1, shelf_id)
    shelf_id, shelf_file = manager.new_shelf()
    shelf_file.close()
    self.assertEqual(2, shelf_id)
    manager.delete_shelf(1)
    shelf_id, shelf_file = manager.new_shelf()
    shelf_file.close()
    self.assertEqual(3, shelf_id)
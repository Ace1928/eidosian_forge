import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def test_invalid_shelf_id(self):
    invalid_id = 'foo'
    err = shelf.InvalidShelfId(invalid_id)
    self.assertEqual('"foo" is not a valid shelf id, try a number instead.', str(err))
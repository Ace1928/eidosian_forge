import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def check_shelve_creation(self, creator, tree):
    self.assertRaises(StopIteration, next, tree.iter_entries_by_dir(specific_files=['foo']))
    s_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
    self.assertEqual(b'foo-id', creator.shelf_transform.final_file_id(s_trans_id))
    self.assertPathDoesNotExist('foo')
    self.assertPathDoesNotExist('bar')
    self.assertShelvedFileEqual('a\n', creator, b'foo-id')
    s_bar_trans_id = creator.shelf_transform.trans_id_file_id(b'bar-id')
    self.assertEqual('directory', creator.shelf_transform.final_kind(s_bar_trans_id))
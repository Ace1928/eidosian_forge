import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def check_shelve_move(self, creator, tree):
    work_trans_id = creator.work_transform.trans_id_file_id(b'baz-id')
    work_foo = creator.work_transform.trans_id_file_id(b'foo-id')
    self.assertEqual(work_foo, creator.work_transform.final_parent(work_trans_id))
    shelf_trans_id = creator.shelf_transform.trans_id_file_id(b'baz-id')
    shelf_bar = creator.shelf_transform.trans_id_file_id(b'bar-id')
    self.assertEqual(shelf_bar, creator.shelf_transform.final_parent(shelf_trans_id))
    creator.transform()
    self.assertEqual('foo/baz', tree.id2path(b'baz-id'))
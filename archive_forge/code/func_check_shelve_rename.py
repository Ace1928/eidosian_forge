import os
from .. import errors, ignores, osutils, shelf, tests, transform, workingtree
from ..bzr import pack
from . import KnownFailure, features
def check_shelve_rename(self, creator):
    work_trans_id = creator.work_transform.trans_id_file_id(b'foo-id')
    self.assertEqual('foo', creator.work_transform.final_name(work_trans_id))
    shelf_trans_id = creator.shelf_transform.trans_id_file_id(b'foo-id')
    self.assertEqual('bar', creator.shelf_transform.final_name(shelf_trans_id))
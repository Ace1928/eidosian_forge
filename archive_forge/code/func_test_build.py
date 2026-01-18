import errno
import os
import sys
import time
from io import BytesIO
from breezy.bzr.transform import resolve_checkout
from breezy.tests.matchers import MatchesTreeChanges
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from ... import osutils, tests, trace, transform, urlutils
from ...bzr.conflicts import (DeletingParent, DuplicateEntry, DuplicateID,
from ...errors import (DuplicateKey, ExistingLimbo, ExistingPendingDeletion,
from ...osutils import file_kind, pathjoin
from ...transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ...transport import FileExists
from ...tree import TreeChange
from .. import TestSkipped, features
from ..features import HardlinkFeature, SymlinkFeature
def test_build(self):
    transform, root = self.transform()
    self.wt.lock_tree_write()
    self.addCleanup(self.wt.unlock)
    self.assertIs(transform.get_tree_parent(root), ROOT_PARENT)
    imaginary_id = transform.trans_id_tree_path('imaginary')
    imaginary_id2 = transform.trans_id_tree_path('imaginary/')
    self.assertEqual(imaginary_id, imaginary_id2)
    self.assertEqual(root, transform.get_tree_parent(imaginary_id))
    self.assertEqual('directory', transform.final_kind(root))
    if self.wt.supports_setting_file_ids():
        self.assertEqual(self.wt.path2id(''), transform.final_file_id(root))
    trans_id = transform.create_path('name', root)
    if self.wt.supports_setting_file_ids():
        self.assertIs(transform.final_file_id(trans_id), None)
    self.assertFalse(transform.final_is_versioned(trans_id))
    self.assertIs(None, transform.final_kind(trans_id))
    transform.create_file([b'contents'], trans_id)
    transform.set_executability(True, trans_id)
    transform.version_file(trans_id, file_id=b'my_pretties')
    self.assertRaises(DuplicateKey, transform.version_file, trans_id, file_id=b'my_pretties')
    if self.wt.supports_setting_file_ids():
        self.assertEqual(transform.final_file_id(trans_id), b'my_pretties')
    self.assertTrue(transform.final_is_versioned(trans_id))
    self.assertEqual(transform.final_parent(trans_id), root)
    self.assertIs(transform.final_parent(root), ROOT_PARENT)
    self.assertIs(transform.get_tree_parent(root), ROOT_PARENT)
    oz_id = transform.create_path('oz', root)
    transform.create_directory(oz_id)
    transform.version_file(oz_id, file_id=b'ozzie')
    trans_id2 = transform.create_path('name2', root)
    transform.create_file([b'contents'], trans_id2)
    transform.set_executability(False, trans_id2)
    transform.version_file(trans_id2, file_id=b'my_pretties2')
    modified_paths = transform.apply().modified_paths
    with self.wt.get_file('name') as f:
        self.assertEqual(b'contents', f.read())
    if self.wt.supports_setting_file_ids():
        self.assertEqual(self.wt.path2id('name'), b'my_pretties')
    self.assertIs(self.wt.is_executable('name'), True)
    self.assertIs(self.wt.is_executable('name2'), False)
    self.assertEqual('directory', file_kind(self.wt.abspath('oz')))
    self.assertEqual(len(modified_paths), 3)
    if self.wt.supports_setting_file_ids():
        tree_mod_paths = [self.wt.abspath(self.wt.id2path(f)) for f in (b'ozzie', b'my_pretties', b'my_pretties2')]
        self.assertSubset(tree_mod_paths, modified_paths)
    transform.finalize()
    transform.finalize()
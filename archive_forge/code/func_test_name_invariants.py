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
def test_name_invariants(self):
    create_tree, root = self.transform()
    root = create_tree.root
    create_tree.new_file('name1', root, [b'hello1'], b'name1')
    create_tree.new_file('name2', root, [b'hello2'], b'name2')
    ddir = create_tree.new_directory('dying_directory', root, b'ddir')
    create_tree.new_file('dying_file', ddir, [b'goodbye1'], b'dfile')
    create_tree.new_file('moving_file', ddir, [b'later1'], b'mfile')
    create_tree.new_file('moving_file2', root, [b'later2'], b'mfile2')
    create_tree.apply()
    mangle_tree, root = self.transform()
    root = mangle_tree.root
    name1 = mangle_tree.trans_id_tree_path('name1')
    name2 = mangle_tree.trans_id_tree_path('name2')
    mangle_tree.adjust_path('name2', root, name1)
    mangle_tree.adjust_path('name1', root, name2)
    ddir = mangle_tree.trans_id_tree_path('dying_directory')
    mangle_tree.delete_contents(ddir)
    dfile = mangle_tree.trans_id_tree_path('dying_directory/dying_file')
    mangle_tree.delete_versioned(dfile)
    mangle_tree.unversion_file(dfile)
    mfile = mangle_tree.trans_id_tree_path('dying_directory/moving_file')
    mangle_tree.adjust_path('mfile', root, mfile)
    newdir = mangle_tree.new_directory('new_directory', root, b'newdir')
    mfile2 = mangle_tree.trans_id_tree_path('moving_file2')
    mangle_tree.adjust_path('mfile2', newdir, mfile2)
    mangle_tree.new_file('newfile', newdir, [b'hello3'], b'dfile')
    if self.wt.supports_setting_file_ids():
        self.assertEqual(mangle_tree.final_file_id(mfile2), b'mfile2')
    self.assertEqual(mangle_tree.final_parent(mfile2), newdir)
    mangle_tree.apply()
    with open(self.wt.abspath('name1')) as f:
        self.assertEqual(f.read(), 'hello2')
    with open(self.wt.abspath('name2')) as f:
        self.assertEqual(f.read(), 'hello1')
    mfile2_path = self.wt.abspath(pathjoin('new_directory', 'mfile2'))
    self.assertEqual(mangle_tree.final_parent(mfile2), newdir)
    with open(mfile2_path) as f:
        self.assertEqual(f.read(), 'later2')
    if self.wt.supports_setting_file_ids():
        self.assertEqual(self.wt.id2path(b'mfile2'), 'new_directory/mfile2')
        self.assertEqual(self.wt.path2id('new_directory/mfile2'), b'mfile2')
    newfile_path = self.wt.abspath(pathjoin('new_directory', 'newfile'))
    with open(newfile_path) as f:
        self.assertEqual(f.read(), 'hello3')
    if self.wt.supports_setting_file_ids():
        self.assertEqual(self.wt.path2id('dying_directory'), b'ddir')
        self.assertIs(self.wt.path2id('dying_directory/dying_file'), None)
    mfile2_path = self.wt.abspath(pathjoin('new_directory', 'mfile2'))
import errno
import os
from io import StringIO
from ... import branch as _mod_branch
from ... import config, controldir, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import bzrdir
from ...bzr.conflicts import ConflictList, ContentsConflict, TextConflict
from ...bzr.inventory import Inventory
from ...bzr.workingtree import InventoryWorkingTree
from ...errors import PathsNotVersionedError, UnsupportedOperation
from ...mutabletree import MutableTree
from ...osutils import getcwd, pathjoin, supports_symlinks
from ...tree import TreeDirectory, TreeFile, TreeLink
from ...workingtree import SettingFileIdUnsupported, WorkingTree
from .. import TestNotApplicable, TestSkipped, features
from . import TestCaseWithWorkingTree
def test_merge_modified(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree_contents([('tree/somefile', b'hello')])
    with tree.lock_write():
        tree.add(['somefile'])
        d = {'somefile': osutils.sha_string(b'hello')}
        if tree.supports_merge_modified():
            tree.set_merge_modified(d)
            mm = tree.merge_modified()
            self.assertEqual(mm, d)
        else:
            self.assertRaises(errors.UnsupportedOperation, tree.set_merge_modified, d)
            mm = tree.merge_modified()
            self.assertEqual(mm, {})
    if tree.supports_merge_modified():
        mm = tree.merge_modified()
        self.assertEqual(mm, d)
    else:
        mm = tree.merge_modified()
        self.assertEqual(mm, {})
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
def test_merge_revert(self):
    from breezy.merge import merge_inner
    this = self.make_branch_and_tree('b1')
    self.build_tree_contents([('b1/a', b'a test\n'), ('b1/b', b'b test\n')])
    this.add(['a', 'b'])
    this.commit(message='')
    base = this.controldir.clone('b2').open_workingtree()
    self.build_tree_contents([('b2/a', b'b test\n')])
    other = this.controldir.clone('b3').open_workingtree()
    self.build_tree_contents([('b3/a', b'c test\n'), ('b3/c', b'c test\n')])
    other.add('c')
    self.build_tree_contents([('b1/b', b'q test\n'), ('b1/d', b'd test\n')])
    this.lock_write()
    self.addCleanup(this.unlock)
    merge_inner(this.branch, other, base, this_tree=this)
    with open('b1/a', 'rb') as a:
        self.assertNotEqual(a.read(), 'a test\n')
    this.revert()
    self.assertFileEqual(b'a test\n', 'b1/a')
    self.assertPathExists('b1/b.~1~')
    if this.supports_merge_modified():
        self.assertPathDoesNotExist('b1/c')
        self.assertPathDoesNotExist('b1/a.~1~')
    else:
        self.assertPathExists('b1/c')
        self.assertPathExists('b1/a.~1~')
    self.assertPathExists('b1/d')
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
def test_basis_tree_returns_last_revision(self):
    wt = self.make_branch_and_tree('.')
    self.build_tree(['foo'])
    wt.add('foo')
    a = wt.commit('A')
    wt.rename_one('foo', 'bar')
    b = wt.commit('B')
    wt.set_parent_ids([b])
    tree = wt.basis_tree()
    tree.lock_read()
    self.assertTrue(tree.has_filename('bar'))
    tree.unlock()
    wt.set_parent_ids([a])
    tree = wt.basis_tree()
    tree.lock_read()
    self.assertTrue(tree.has_filename('foo'))
    tree.unlock()
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
def test_filter_unversioned_files(self):
    tree = self.make_branch_and_tree('.')
    paths = ['here-and-versioned', 'here-and-not-versioned', 'not-here-and-versioned', 'not-here-and-not-versioned']
    tree.add(['here-and-versioned', 'not-here-and-versioned'], kinds=['file', 'file'])
    self.build_tree(['here-and-versioned', 'here-and-not-versioned'])
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual({'not-here-and-not-versioned', 'here-and-not-versioned'}, tree.filter_unversioned_files(paths))
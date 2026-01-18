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
def test__write_inventory(self):
    tree = self.make_branch_and_tree('.')
    if not isinstance(tree, InventoryWorkingTree):
        raise TestNotApplicable('_write_inventory does not exist on non-inventory working trees')
    self.build_tree(['present', 'unknown'])
    inventory = Inventory(tree.path2id(''))
    inventory.add_path('missing', 'file', b'missing-id')
    inventory.add_path('present', 'file', b'present-id')
    tree.lock_write()
    tree._write_inventory(inventory)
    tree.unlock()
    with tree.lock_read():
        present_stat = os.lstat('present')
        unknown_stat = os.lstat('unknown')
        expected_results = [('', [('missing', 'missing', 'unknown', None, 'file'), ('present', 'present', 'file', present_stat, 'file'), ('unknown', 'unknown', 'file', unknown_stat, None)])]
        self.assertEqual(expected_results, list(tree.walkdirs()))
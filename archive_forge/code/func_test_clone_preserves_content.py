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
def test_clone_preserves_content(self):
    wt = self.make_branch_and_tree('source')
    self.build_tree(['added', 'deleted', 'notadded'], transport=wt.controldir.transport.clone('..'))
    wt.add('deleted')
    wt.commit('add deleted')
    wt.remove('deleted')
    wt.add('added')
    cloned_dir = wt.controldir.clone('target')
    cloned = cloned_dir.open_workingtree()
    cloned_transport = cloned.controldir.transport.clone('..')
    self.assertFalse(cloned_transport.has('deleted'))
    self.assertTrue(cloned_transport.has('added'))
    self.assertFalse(cloned_transport.has('notadded'))
    self.assertTrue(cloned.is_versioned('added'))
    self.assertFalse(cloned.is_versioned('deleted'))
    self.assertFalse(cloned.is_versioned('notadded'))
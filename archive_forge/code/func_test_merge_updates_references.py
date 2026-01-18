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
def test_merge_updates_references(self):
    orig_tree = self.make_tree_with_reference('branch', 'reference')
    tree = orig_tree.controldir.sprout('tree').open_workingtree()
    tree.commit('foo')
    orig_tree.pull(tree.branch)
    checkout = orig_tree.branch.create_checkout('checkout', lightweight=True)
    checkout.commit('bar')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    merger = merge.Merger.from_revision_ids(tree, orig_tree.branch.last_revision(), other_branch=orig_tree.branch)
    merger.merge_type = merge.Merge3Merger
    merger.do_merge()
    self.assertEqual(urlutils.join(urlutils.strip_segment_parameters(orig_tree.branch.user_url), 'reference'), urlutils.join(urlutils.strip_segment_parameters(tree.branch.user_url), tree.get_reference_info('path/to/file')))
import os
import stat
from dulwich import __version__ as dulwich_version
from dulwich.diff_tree import RenameDetector, tree_changes
from dulwich.index import IndexEntry, ConflictedIndexEntry
from dulwich.object_store import OverlayObjectStore
from dulwich.objects import S_IFGITLINK, ZERO_SHA, Blob, Tree
from ... import conflicts as _mod_conflicts
from ... import workingtree as _mod_workingtree
from ...bzr.inventorytree import InventoryTreeChange as TreeChange
from ...delta import TreeDelta
from ...tests import TestCase, TestCaseWithTransport
from ..mapping import default_mapping
from ..tree import tree_delta_from_git_changes
def expectDelta(self, expected_changes, expected_extras=None, want_unversioned=False, tree_id=None, rename_detector=None):
    if tree_id is None:
        try:
            tree_id = self.store[self.wt.branch.repository._git.head()].tree
        except KeyError:
            tree_id = None
    with self.wt.lock_read():
        changes, extras = changes_between_git_tree_and_working_copy(self.store, tree_id, self.wt, want_unversioned=want_unversioned, rename_detector=rename_detector)
        self.assertEqual(expected_changes, list(changes))
    if expected_extras is None:
        expected_extras = set()
    self.assertEqual(set(expected_extras), set(extras))
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
def test_is_ignored_directory(self):
    self.assertFalse(self.tree.is_ignored('a'))
    self.build_tree(['a/'])
    self.assertFalse(self.tree.is_ignored('a'))
    self.build_tree_contents([('.gitignore', 'a\n')])
    self.tree._ignoremanager = None
    self.assertTrue(self.tree.is_ignored('a'))
    self.build_tree_contents([('.gitignore', 'a/\n')])
    self.tree._ignoremanager = None
    self.assertTrue(self.tree.is_ignored('a'))
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
def test_missing_added_file(self):
    self.build_tree(['a'])
    self.wt.add(['a'])
    os.unlink('a')
    a = Blob.from_string(b'contents of a\n')
    t = Tree()
    t.add(b'a', 0, ZERO_SHA)
    self.expectDelta([('add', (None, None, None), (b'', stat.S_IFDIR, t.id)), ('add', (None, None, None), (b'a', 0, ZERO_SHA))], [])
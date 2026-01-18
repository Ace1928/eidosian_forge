import stat
from dulwich.objects import Blob, Tree
from breezy.bzr.inventorytree import InventoryTreeChange as TreeChange
from breezy.delta import TreeDelta
from breezy.errors import PathsNotVersionedError
from breezy.git.mapping import default_mapping
from breezy.git.tree import (changes_from_git_changes,
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_kind_change(self):
    a = Blob.from_string(b'a')
    b = Blob.from_string(b'target')
    delta = self.transform([('modify', (b'a', stat.S_IFREG | 420, a), (b'a', stat.S_IFLNK, b))])
    expected_delta = TreeDelta()
    expected_delta.kind_changed.append(TreeChange(b'git:a', ('a', 'a'), True, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('a', 'a'), ('file', 'symlink'), (False, False), False))
    self.assertEqual(expected_delta, delta)
import stat
from dulwich.objects import Blob, Tree
from breezy.bzr.inventorytree import InventoryTreeChange as TreeChange
from breezy.delta import TreeDelta
from breezy.errors import PathsNotVersionedError
from breezy.git.mapping import default_mapping
from breezy.git.tree import (changes_from_git_changes,
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.workingtree import WorkingTree
def test_empty_directory(self):
    self.make_branch_and_tree('t1', format='git')
    wt = WorkingTree.open('t1')
    self.build_tree(['t1/dir/'])
    wt.add(['dir'])
    self.assertEqual(['dir'], list(wt.find_related_paths_across_trees(['dir'])))
    self.assertRaises(PathsNotVersionedError, wt.find_related_paths_across_trees, ['dir/file'])
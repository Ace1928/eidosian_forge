from breezy import osutils, tests
from breezy.git.branch import GitBranch
from breezy.mutabletree import MutableTree
from breezy.tests import TestSkipped, features, per_tree
from breezy.transform import PreviewTree
def test_kind_symlink(self):
    self.assertEqual('symlink', self.tree.kind('symlink'))
    self.assertIs(None, self.tree.get_file_size('symlink'))
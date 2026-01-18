from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_has_versioned_directories(self):
    work_tree = self.make_branch_and_tree('tree')
    tree = self._convert_tree(work_tree)
    self.assertIn(tree.has_versioned_directories(), (True, False))
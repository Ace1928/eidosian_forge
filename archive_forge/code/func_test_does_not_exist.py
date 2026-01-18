from breezy import conflicts, errors, osutils, revisiontree, tests
from breezy import transport as _mod_transport
from breezy.bzr import workingtree_4
from breezy.tests import TestSkipped
from breezy.tests.per_tree import TestCaseWithTree
from breezy.tree import MissingNestedTree
def test_does_not_exist(self):
    work_tree = self.make_branch_and_tree('.')
    self.build_tree(['a/'])
    work_tree.add(['a'])
    tree = self._convert_tree(work_tree)
    self.assertRaises(_mod_transport.NoSuchFile, lambda: list(tree.iter_child_entries('unknown')))
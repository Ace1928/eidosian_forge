from breezy import errors
from breezy.bzr.inventorytree import InventoryTree
from breezy.tests import TestNotApplicable, features
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_find_single_root(self):
    tree = self.make_branch_and_tree('tree')
    if not isinstance(tree, InventoryTree):
        raise TestNotApplicable('test not applicable on non-inventory tests')
    self.assertExpectedIds([tree.path2id('')], tree, [''])
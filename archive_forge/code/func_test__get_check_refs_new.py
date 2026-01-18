from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test__get_check_refs_new(self):
    tree = self.make_branch_and_tree('tree')
    if not isinstance(tree, InventoryWorkingTree):
        raise TestNotApplicable('_get_check_refs only relevant for inventory working trees')
    self.assertEqual({('trees', b'null:')}, set(tree._get_check_refs()))
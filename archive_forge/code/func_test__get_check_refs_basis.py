from breezy.bzr.workingtree import InventoryWorkingTree
from breezy.tests import TestNotApplicable
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test__get_check_refs_basis(self):
    tree = self.make_branch_and_tree('tree')
    if not isinstance(tree, InventoryWorkingTree):
        raise TestNotApplicable('_get_check_refs only relevant for inventory working trees')
    revid = tree.commit('first post')
    self.assertEqual({('trees', revid)}, set(tree._get_check_refs()))
from breezy.tests.matchers import *
from breezy.tests.per_tree import TestCaseWithTree
def test_lock_read(self):
    work_tree = self.make_branch_and_tree('wt')
    tree = self.workingtree_to_test_tree(work_tree)
    self.assertThat(tree.lock_read, ReturnsUnlockable(tree))
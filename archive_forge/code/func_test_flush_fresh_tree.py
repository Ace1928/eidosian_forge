import sys
from breezy import errors
from breezy.tests import TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_flush_fresh_tree(self):
    tree = self.make_branch_and_tree('t1')
    with tree.lock_write():
        tree.flush()
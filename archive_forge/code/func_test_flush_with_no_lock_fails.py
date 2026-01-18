import sys
from breezy import errors
from breezy.tests import TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_flush_with_no_lock_fails(self):
    tree = self.make_branch_and_tree('t1')
    self.assertRaises(errors.NotWriteLocked, tree.flush)
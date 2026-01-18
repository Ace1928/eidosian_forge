import sys
from breezy import errors
from breezy.tests import TestSkipped
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_flush_with_read_lock_fails(self):
    """Flush cannot be used during a read lock."""
    tree = self.make_branch_and_tree('t1')
    with tree.lock_read():
        self.assertRaises(errors.NotWriteLocked, tree.flush)
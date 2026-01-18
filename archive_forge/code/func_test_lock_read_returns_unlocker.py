import sys
from breezy import branch, errors
from breezy.tests import TestSkipped
from breezy.tests.matchers import *
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
def test_lock_read_returns_unlocker(self):
    wt = self.make_branch_and_tree('.')
    self.assertThat(wt.lock_read, ReturnsUnlockable(wt))
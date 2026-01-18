from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_hooks_no_branch(self):
    self.run_bzr('hooks')
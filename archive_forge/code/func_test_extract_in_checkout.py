from .. import branch, errors
from . import TestCaseWithTransport
def test_extract_in_checkout(self):
    a_branch = self.make_branch('branch', format='rich-root-pack')
    self.extract_in_checkout(a_branch)
    b_branch = branch.Branch.open('branch/b')
    b_branch_ref = branch.Branch.open('a/b')
    self.assertEqual(b_branch.base, b_branch_ref.base)
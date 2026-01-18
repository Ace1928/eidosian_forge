from .. import branch, errors
from . import TestCaseWithTransport
def test_good_repo_format(self):
    repo = self.make_repository('branch', shared=True, format='dirstate-with-subtree')
    a_branch = repo.controldir.create_branch()
    wt_b = self.extract_in_checkout(a_branch)
    self.assertEqual(wt_b.branch.repository.controldir.transport.base, repo.controldir.transport.base)
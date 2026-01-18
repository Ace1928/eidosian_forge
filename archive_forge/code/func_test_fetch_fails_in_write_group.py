from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_fails_in_write_group(self):
    repo = self.make_repository('a')
    repo.lock_write()
    self.addCleanup(repo.unlock)
    repo.start_write_group()
    self.addCleanup(repo.abort_write_group)
    self.assertRaises(errors.BzrError, repo.fetch, repo)
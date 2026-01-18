from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_revision_already_exists(self):
    source_repo, rev1 = self.make_repository_with_one_revision()
    target_repo = self.make_repository('target')
    target_repo.fetch(source_repo, revision_id=rev1)
    target_repo.fetch(source_repo, revision_id=rev1)
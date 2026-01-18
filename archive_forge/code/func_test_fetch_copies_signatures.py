from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_copies_signatures(self):
    source_repo, rev1 = self.makeARepoWithSignatures()
    target_repo = self.make_repository('target')
    target_repo.fetch(source_repo, revision_id=None)
    self.assertEqual(source_repo.get_signature_text(rev1), target_repo.get_signature_text(rev1))
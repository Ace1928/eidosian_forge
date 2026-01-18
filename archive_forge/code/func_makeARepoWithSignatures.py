from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def makeARepoWithSignatures(self):
    wt = self.make_branch_and_tree('a-repo-with-sigs')
    rev1 = wt.commit('rev1', allow_pointless=True)
    repo = wt.branch.repository
    repo.lock_write()
    repo.start_write_group()
    try:
        repo.sign_revision(rev1, gpg.LoopbackGPGStrategy(None))
    except errors.UnsupportedOperation:
        self.assertFalse(repo._format.supports_revision_signatures)
        raise TestNotApplicable('repository format does not support signatures')
    repo.commit_write_group()
    repo.unlock()
    return (repo, rev1)
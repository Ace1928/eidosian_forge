from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def make_repository_with_one_revision(self):
    wt = self.make_branch_and_tree('source')
    rev1 = wt.commit('rev1', allow_pointless=True)
    return (wt.branch.repository, rev1)
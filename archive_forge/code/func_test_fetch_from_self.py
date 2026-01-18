from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_from_self(self):
    tree = self.make_branch_and_tree('.')
    rev_id = tree.commit('one')
    repo = tree.branch.repository.controldir.open_repository()
    tree.branch.repository.fetch(repo, rev_id)
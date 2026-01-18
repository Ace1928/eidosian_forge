from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_fetch_copies_from_stacked_on_and_stacked(self):
    stacked, unstacked, rev1 = self.prepare_stacked_on_fetch()
    tree = stacked.branch.create_checkout('local')
    rev2 = tree.commit('second commit')
    unstacked.fetch(stacked.branch.repository, rev2)
    unstacked.get_revision(rev1)
    unstacked.get_revision(rev2)
    self.check_lines_added_or_present(stacked.branch, rev1)
    self.check_lines_added_or_present(stacked.branch, rev2)
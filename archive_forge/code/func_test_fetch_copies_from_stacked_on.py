from breezy import branch as _mod_branch
from breezy import check, controldir, errors
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable, fixtures, transport_util
from breezy.tests.per_branch import TestCaseWithBranch
def test_fetch_copies_from_stacked_on(self):
    stacked, unstacked, rev1 = self.prepare_stacked_on_fetch()
    unstacked.fetch(stacked.branch.repository, rev1)
    unstacked.get_revision(rev1)
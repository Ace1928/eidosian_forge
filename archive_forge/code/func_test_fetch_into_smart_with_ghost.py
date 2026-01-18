from breezy import controldir, errors, gpg, repository
from breezy.bzr import remote
from breezy.bzr.inventory import ROOT_ID
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.tests.per_repository import TestCaseWithRepository
def test_fetch_into_smart_with_ghost(self):
    trans = self.make_smart_server('target')
    source_b, b_revid = self.make_simple_branch_with_ghost()
    if not source_b.controldir._format.supports_transport(trans):
        raise TestNotApplicable('format does not support transport')
    target = self.make_repository('target')
    target = repository.Repository.open(trans.base)
    target.lock_write()
    self.addCleanup(target.unlock)
    try:
        target.fetch(source_b.repository, revision_id=b_revid)
    except errors.TokenLockingNotSupported:
        target.lock_read()
        self.knownFailure('some repositories fail to fetch via the smart server because of locking issues.')
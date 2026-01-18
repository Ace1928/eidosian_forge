import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def test_no_upgrade_recommendation_from_bzrdir(self):
    self.run_bzr('init --format=knit a')
    out, err = self.run_bzr('revno a')
    if err.find('upgrade') > -1:
        self.fail("message shouldn't suggest upgrade:\n%s" % err)
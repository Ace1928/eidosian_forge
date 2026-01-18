import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def assertLegalOption(self, option_str):
    self.run_bzr('init --format=pack-0.92 branch-foo')
    self.run_bzr('upgrade --format=2a branch-foo {}'.format(option_str))
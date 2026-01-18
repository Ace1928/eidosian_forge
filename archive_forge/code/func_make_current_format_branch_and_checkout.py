import os
import stat
from breezy import bzr, controldir, lockdir, ui, urlutils
from breezy.bzr import bzrdir
from breezy.bzr.knitpack_repo import RepositoryFormatKnitPack1
from breezy.tests import TestCaseWithTransport, features
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
def make_current_format_branch_and_checkout(self):
    current_tree = self.make_branch_and_tree('current_format_branch', format='default')
    current_tree.branch.create_checkout(self.get_url('current_format_checkout'), lightweight=True)
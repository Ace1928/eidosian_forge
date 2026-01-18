import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_init_existing_branch(self):
    self.make_branch('.')
    out, err = self.run_bzr_error(['Already a branch'], ['init', self.get_url()])
    self.assertFalse(re.search('use brz checkout', err))
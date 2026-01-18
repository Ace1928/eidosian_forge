import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_init_colocated(self):
    """Smoke test for constructing a colocated branch."""
    out, err = self.run_bzr('init --format=development-colo file:,branch=abranch')
    self.assertEqual('Created a standalone tree (format: development-colo)\n', out)
    self.assertEqual('', err)
    out, err = self.run_bzr('branches')
    self.assertEqual('  abranch\n', out)
    self.assertEqual('', err)
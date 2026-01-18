import os
import sys
from ... import (branch, controldir, errors, repository, upgrade, urlutils,
from ...bzr import bzrdir
from ...bzr.tests import test_bundle
from ...osutils import getcwd
from ...tests import TestCaseWithTransport
from ...tests.test_sftp_transport import TestCaseWithSFTPServer
from .branch import BzrBranchFormat4
from .bzrdir import BzrDirFormat5, BzrDirFormat6
def test_break_lock_format_5_bzrdir(self):
    self.make_branch_and_tree('foo', format=BzrDirFormat5())
    out, err = self.run_bzr('break-lock foo')
    self.assertEqual('', out)
    self.assertEqual('', err)
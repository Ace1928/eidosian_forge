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
def test_needs_conversion(self):
    dir = BzrDirFormat6().initialize(self.get_url())
    self.assertTrue(dir.needs_format_conversion(bzrdir.BzrDirFormat.get_default_format()))
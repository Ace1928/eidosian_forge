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
def test_info_locking_oslocks(self):
    if sys.platform == 'win32':
        self.skip("don't use oslocks on win32 in unix manner")
    self.thisFailsStrictLockCheck()
    tree = self.make_branch_and_tree('branch', format=BzrDirFormat6())
    out, err = self.run_bzr('info -v branch')
    self.assertEqualDiff('Standalone tree (format: weave)\nLocation:\n  branch root: {}\n\nFormat:\n       control: All-in-one format 6\n  working tree: Working tree format 2\n        branch: Branch format 4\n    repository: {}\n\nIn the working tree:\n         0 unchanged\n         0 modified\n         0 added\n         0 removed\n         0 renamed\n         0 copied\n         0 unknown\n         0 ignored\n         0 versioned subdirectories\n\nBranch history:\n         0 revisions\n\nRepository:\n         0 revisions\n'.format('branch', tree.branch.repository._format.get_format_description()), out)
    self.assertEqual('', err)
    tree.lock_write()
    out, err = self.run_bzr('info -v branch')
    self.assertEqualDiff('Standalone tree (format: weave)\nLocation:\n  branch root: {}\n\nFormat:\n       control: All-in-one format 6\n  working tree: Working tree format 2\n        branch: Branch format 4\n    repository: {}\n\nIn the working tree:\n         0 unchanged\n         0 modified\n         0 added\n         0 removed\n         0 renamed\n         0 copied\n         0 unknown\n         0 ignored\n         0 versioned subdirectories\n\nBranch history:\n         0 revisions\n\nRepository:\n         0 revisions\n'.format('branch', tree.branch.repository._format.get_format_description()), out)
    self.assertEqual('', err)
    tree.unlock()
import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_init_existing_without_workingtree(self):
    repo = self.make_repository('.', shared=True)
    repo.set_make_working_trees(False)
    self.run_bzr('init subdir')
    out, err = self.run_bzr('init subdir', retcode=3)
    self.assertContainsRe(err, 'ontains a branch.*but no working tree.*checkout')
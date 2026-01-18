import os
import re
from breezy import branch as _mod_branch
from breezy import config as _mod_config
from breezy import osutils, urlutils
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_init_branch(self):
    out, err = self.run_bzr('init')
    self.assertEqual('Created a standalone tree (format: {})\n'.format(self._default_label), out)
    self.assertEqual('', err)
    out, err = self.run_bzr('init subdir1')
    self.assertEqual('Created a standalone tree (format: {})\n'.format(self._default_label), out)
    self.assertEqual('', err)
    WorkingTree.open('subdir1')
    self.run_bzr_error(['Parent directory of subdir2/nothere does not exist'], 'init subdir2/nothere')
    out, err = self.run_bzr('init subdir2/nothere', retcode=3)
    self.assertEqual('', out)
    os.mkdir('subdir2')
    out, err = self.run_bzr('init subdir2')
    self.assertEqual('Created a standalone tree (format: {})\n'.format(self._default_label), out)
    self.assertEqual('', err)
    out, err = self.run_bzr('init subdir2', retcode=3)
    self.assertEqual('', out)
    self.assertTrue(err.startswith('brz: ERROR: Already a branch:'))
import os
import sys
from breezy import urlutils
from breezy.branch import Branch
from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_new_files(self):
    if sys.platform == 'win32':
        raise TestSkipped('chmod has no effect on win32')
    _t = self.get_transport()
    os.mkdir('local')
    t_local = self.make_branch_and_tree('local')
    b_local = t_local.branch
    with open('local/a', 'wb') as f:
        f.write(b'foo\n')
    t_local.add('a')
    t_local.commit('foo')
    chmod_r('local/.bzr', 420, 493)
    check_mode_r(self, 'local/.bzr', 420, 493)
    t = WorkingTree.open('local')
    b_local = t.branch
    self.assertEqualMode(493, b_local.control_files._dir_mode)
    self.assertEqualMode(420, b_local.control_files._file_mode)
    self.assertEqualMode(493, b_local.controldir._get_dir_mode())
    self.assertEqualMode(420, b_local.controldir._get_file_mode())
    os.mkdir('sftp')
    sftp_url = self.get_url('sftp')
    b_sftp = ControlDir.create_branch_and_repo(sftp_url)
    b_sftp.pull(b_local)
    del b_sftp
    chmod_r('sftp/.bzr', 420, 493)
    check_mode_r(self, 'sftp/.bzr', 420, 493)
    b_sftp = Branch.open(sftp_url)
    self.assertEqualMode(493, b_sftp.control_files._dir_mode)
    self.assertEqualMode(420, b_sftp.control_files._file_mode)
    self.assertEqualMode(493, b_sftp.controldir._get_dir_mode())
    self.assertEqualMode(420, b_sftp.controldir._get_file_mode())
    with open('local/a', 'wb') as f:
        f.write(b'foo2\n')
    t_local.commit('foo2')
    b_sftp.pull(b_local)
    check_mode_r(self, 'sftp/.bzr', 420, 493)
    with open('local/b', 'wb') as f:
        f.write(b'new b\n')
    t_local.add('b')
    t_local.commit('new b')
    b_sftp.pull(b_local)
    check_mode_r(self, 'sftp/.bzr', 420, 493)
    del b_sftp
    chmod_r('sftp/.bzr', 436, 509)
    check_mode_r(self, 'sftp/.bzr', 436, 509)
    b_sftp = Branch.open(sftp_url)
    self.assertEqualMode(509, b_sftp.control_files._dir_mode)
    self.assertEqualMode(436, b_sftp.control_files._file_mode)
    self.assertEqualMode(509, b_sftp.controldir._get_dir_mode())
    self.assertEqualMode(436, b_sftp.controldir._get_file_mode())
    with open('local/a', 'wb') as f:
        f.write(b'foo3\n')
    t_local.commit('foo3')
    b_sftp.pull(b_local)
    check_mode_r(self, 'sftp/.bzr', 436, 509)
    with open('local/c', 'wb') as f:
        f.write(b'new c\n')
    t_local.add('c')
    t_local.commit('new c')
    b_sftp.pull(b_local)
    check_mode_r(self, 'sftp/.bzr', 436, 509)
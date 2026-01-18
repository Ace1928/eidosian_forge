import os
import sys
from breezy import urlutils
from breezy.branch import Branch
from breezy.controldir import ControlDir
from breezy.tests import TestCaseWithTransport, TestSkipped
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_new_files_group_sticky_bit(self):
    if sys.platform == 'win32':
        raise TestSkipped('chmod has no effect on win32')
    elif sys.platform == 'darwin' or 'freebsd' in sys.platform:
        os.chown(self.test_dir, os.getuid(), os.getgid())
    t = self.make_branch_and_tree('.')
    b = t.branch
    chmod_r('.bzr', 436, 1533)
    check_mode_r(self, '.bzr', 436, 1533)
    t = WorkingTree.open('.')
    b = t.branch
    self.assertEqualMode(1533, b.control_files._dir_mode)
    self.assertEqualMode(436, b.control_files._file_mode)
    self.assertEqualMode(1533, b.controldir._get_dir_mode())
    self.assertEqualMode(436, b.controldir._get_file_mode())
    with open('a', 'wb') as f:
        f.write(b'foo4\n')
    t.commit('foo4')
    check_mode_r(self, '.bzr', 436, 1533)
    with open('d', 'wb') as f:
        f.write(b'new d\n')
    t.add('d')
    t.commit('new d')
    check_mode_r(self, '.bzr', 436, 1533)
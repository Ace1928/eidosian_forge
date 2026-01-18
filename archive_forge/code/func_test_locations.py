import os
import re
import sys
import breezy
from breezy import osutils
from breezy.branch import Branch
from breezy.errors import CommandError
from breezy.tests import TestCaseWithTransport
from breezy.tests.http_utils import TestCaseWithWebserver
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_locations(self):
    """Using and remembering different locations"""
    os.mkdir('a')
    os.chdir('a')
    self.run_bzr('init')
    self.run_bzr('commit -m unchanged --unchanged')
    self.run_bzr('pull', retcode=3)
    self.run_bzr('merge', retcode=3)
    self.run_bzr('branch . ../b')
    os.chdir('../b')
    self.run_bzr('pull')
    self.run_bzr('branch . ../c')
    self.run_bzr('pull ../c')
    self.run_bzr('merge')
    os.chdir('../a')
    self.run_bzr('pull ../b')
    self.run_bzr('pull')
    self.run_bzr('pull ../c')
    self.run_bzr('branch ../c ../d')
    osutils.rmtree('../c')
    self.run_bzr('pull')
    os.chdir('../b')
    self.run_bzr('pull')
    os.chdir('../d')
    self.run_bzr('pull', retcode=3)
    self.run_bzr('pull ../a --remember')
    self.run_bzr('pull')
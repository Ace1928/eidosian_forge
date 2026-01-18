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
def create_conflicts(self):
    """Create a conflicted tree"""
    os.mkdir('base')
    os.chdir('base')
    with open('hello', 'wb') as f:
        f.write(b'hi world')
    with open('answer', 'wb') as f:
        f.write(b'42')
    self.run_bzr('init')
    self.run_bzr('add')
    self.run_bzr('commit -m base')
    self.run_bzr('branch . ../other')
    self.run_bzr('branch . ../this')
    os.chdir('../other')
    with open('hello', 'wb') as f:
        f.write(b'Hello.')
    with open('answer', 'wb') as f:
        f.write(b'Is anyone there?')
    self.run_bzr('commit -m other')
    os.chdir('../this')
    with open('hello', 'wb') as f:
        f.write(b'Hello, world')
    self.run_bzr('mv answer question')
    with open('question', 'wb') as f:
        f.write(b'What do you get when you multiply sixtimes nine?')
    self.run_bzr('commit -m this')
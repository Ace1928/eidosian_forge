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
def example_branch(test):
    test.run_bzr('init')
    with open('hello', 'w') as f:
        f.write('foo')
    test.run_bzr('add hello')
    test.run_bzr('commit -m setup hello')
    with open('goodbye', 'w') as f:
        f.write('baz')
    test.run_bzr('add goodbye')
    test.run_bzr('commit -m setup goodbye')
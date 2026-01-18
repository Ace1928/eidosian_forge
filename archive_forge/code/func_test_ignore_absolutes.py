import os
import re
import breezy
from breezy import ignores, osutils
from breezy.branch import Branch
from breezy.errors import CommandError
from breezy.osutils import pathjoin
from breezy.tests import TestCaseWithTransport
from breezy.tests.test_sftp_transport import TestCaseWithSFTPServer
from breezy.workingtree import WorkingTree
def test_ignore_absolutes(self):
    """'ignore' with an absolute path returns an error"""
    self.make_branch_and_tree('.')
    self.run_bzr_error(('brz: ERROR: NAME_PATTERN should not be an absolute path\n',), 'ignore /crud')
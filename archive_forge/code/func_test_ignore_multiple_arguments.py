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
def test_ignore_multiple_arguments(self):
    """'ignore' works with multiple arguments"""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b', 'c', 'd'])
    self.assertEqual(list(tree.unknowns()), ['a', 'b', 'c', 'd'])
    self.run_bzr('ignore a b c')
    self.assertEqual(list(tree.unknowns()), ['d'])
    self.check_file_contents('.bzrignore', b'a\nb\nc\n')
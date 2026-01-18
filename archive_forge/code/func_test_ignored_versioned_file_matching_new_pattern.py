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
def test_ignored_versioned_file_matching_new_pattern(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['a', 'b'])
    tree.add(['a', 'b'])
    self.run_bzr('ignore *')
    out, err = self.run_bzr('ignore b')
    self.assertEqual(out, "Warning: the following files are version controlled and match your ignore pattern:\nb\nThese files will continue to be version controlled unless you 'brz remove' them.\n")
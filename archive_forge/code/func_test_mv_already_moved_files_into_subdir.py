import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_files_into_subdir(self):
    """Test brz mv original_files to versioned_directory.

        Tests if files which has already been moved into a versioned
        directory by an external tool, is handled correctly by brz mv.
        Setup: a1, a2, sub are in the working tree.
        User does: mv a1 sub/.; brz mv a1 a2 sub
        """
    self.build_tree(['a1', 'a2', 'sub/'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a1', 'a2', 'sub'])
    osutils.rename('a1', 'sub/a1')
    self.run_bzr('mv a1 a2 sub')
    self.assertMoved('a1', 'sub/a1')
    self.assertMoved('a2', 'sub/a2')
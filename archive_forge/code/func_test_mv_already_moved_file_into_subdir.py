import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_file_into_subdir(self):
    """Test brz mv original_file to versioned_directory/file.

        Tests if a file which has already been moved into a versioned
        directory by an external tool, is handled correctly by brz mv.
        Setup: a and sub/ are in the working tree.
        User does: mv a sub/a; brz mv a sub/a
        """
    self.build_tree(['a', 'sub/'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a', 'sub'])
    osutils.rename('a', 'sub/a')
    self.run_bzr('mv a sub/a')
    self.assertMoved('a', 'sub/a')
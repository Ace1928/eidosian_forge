import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_files_forcing_after(self):
    """Test brz mv versioned_files to directory/unversioned_file.

        Tests if an attempt to move an existing versioned file to an existing
        unversioned file in some other directory will fail, informing the user
        to use the --after option to force this.

        Setup: a1, a2, sub are versioned and in the working tree,
               sub/a1, sub/a2 are in working tree.
        User does: mv a* sub; touch a1; touch a2; brz mv a1 a2 sub
        """
    self.build_tree(['a1', 'a2', 'sub/', 'sub/a1', 'sub/a2'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a1', 'a2', 'sub'])
    osutils.rename('a1', 'sub/a1')
    osutils.rename('a2', 'sub/a2')
    self.build_tree(['a1'])
    self.build_tree(['a2'])
    self.run_bzr_error(['^brz: ERROR: Could not rename a1 => sub/a1 because both files exist. \\(Use --after to tell brz about a rename that has already happened\\)$'], 'mv a1 a2 sub')
    self.assertPathExists('a1')
    self.assertPathExists('a2')
    self.assertPathExists('sub/a1')
    self.assertPathExists('sub/a2')
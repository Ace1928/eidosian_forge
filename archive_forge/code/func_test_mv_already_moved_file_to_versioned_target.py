import os
import breezy.branch
from breezy import osutils, workingtree
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import (CaseInsensitiveFilesystemFeature,
def test_mv_already_moved_file_to_versioned_target(self):
    """Test brz mv existing_file to versioned_file.

        Tests if an attempt to move an existing versioned file
        to another versiond file will fail.
        Setup: a and b are in the working tree.
        User does: rm b; mv a b; brz mv a b
        """
    self.build_tree(['a', 'b'])
    tree = self.make_branch_and_tree('.')
    tree.add(['a', 'b'])
    os.remove('b')
    osutils.rename('a', 'b')
    self.run_bzr_error(['^brz: ERROR: Could not move a => b. b is already versioned\\.$'], 'mv a b')
    self.assertPathDoesNotExist('a')
    self.assertPathExists('b')
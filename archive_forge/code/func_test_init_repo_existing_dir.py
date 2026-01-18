import os
import breezy.errors as errors
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.controldir import ControlDir
from breezy.tests import TestCaseInTempDir
def test_init_repo_existing_dir(self):
    """Make repo in existing directory.

        (Malone #38331)
        """
    out, err = self.run_bzr('init-shared-repository .')
    dir = ControlDir.open('.')
    self.assertTrue(dir.open_repository())
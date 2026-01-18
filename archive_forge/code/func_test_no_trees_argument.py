import os
import breezy.errors as errors
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.controldir import ControlDir
from breezy.tests import TestCaseInTempDir
def test_no_trees_argument(self):
    self.run_bzr('init-shared-repo --no-trees notrees')
    repo = ControlDir.open('notrees').open_repository()
    self.assertEqual(False, repo.make_working_trees())
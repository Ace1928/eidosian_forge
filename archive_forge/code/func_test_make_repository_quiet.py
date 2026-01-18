import os
import breezy.errors as errors
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.controldir import ControlDir
from breezy.tests import TestCaseInTempDir
def test_make_repository_quiet(self):
    out, err = self.run_bzr('init-shared-repository a -q')
    self.assertEqual(out, '')
    self.assertEqual(err, '')
    dir = ControlDir.open('a')
    self.assertIs(dir.open_repository().is_shared(), True)
    self.assertRaises(errors.NotBranchError, dir.open_branch)
    self.assertRaises(errors.NoWorkingTree, dir.open_workingtree)
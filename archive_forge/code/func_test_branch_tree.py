import os
import breezy.errors as errors
from breezy.bzr.bzrdir import BzrDirMetaFormat1
from breezy.controldir import ControlDir
from breezy.tests import TestCaseInTempDir
def test_branch_tree(self):
    self.run_bzr('init-shared-repo --trees a')
    self.run_bzr('init --format=default b')
    with open('b/hello', 'w') as f:
        f.write('bar')
    self.run_bzr('add b/hello')
    self.run_bzr('commit -m bar b/hello')
    self.run_bzr('branch b a/c')
    cdir = ControlDir.open('a/c')
    cdir.open_branch()
    self.assertRaises(errors.NoRepositoryPresent, cdir.open_repository)
    self.assertPathExists('a/c/hello')
    cdir.open_workingtree()
import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
def test_checkout_reconstitutes_working_trees(self):
    os.mkdir('treeless-branch')
    branch = controldir.ControlDir.create_branch_convenience('treeless-branch', force_new_tree=False, format=bzrdir.BzrDirMetaFormat1())
    self.assertRaises(errors.NoWorkingTree, branch.controldir.open_workingtree)
    out, err = self.run_bzr('checkout treeless-branch')
    branch.controldir.open_workingtree()
    out, err = self.run_bzr('diff treeless-branch')
    branch = controldir.ControlDir.create_branch_convenience('.', force_new_tree=False, format=bzrdir.BzrDirMetaFormat1())
    self.assertRaises(errors.NoWorkingTree, branch.controldir.open_workingtree)
    out, err = self.run_bzr('checkout')
    branch.controldir.open_workingtree()
    out, err = self.run_bzr('diff')
import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
def test_checkout_in_branch_with_r(self):
    branch = _mod_branch.Branch.open('branch')
    branch.controldir.destroy_workingtree()
    self.run_bzr('checkout -r 1', working_dir='branch')
    tree = workingtree.WorkingTree.open('branch')
    self.assertEqual(self.rev1, tree.last_revision())
    branch.controldir.destroy_workingtree()
    self.run_bzr('checkout -r 0', working_dir='branch')
    self.assertEqual(b'null:', tree.last_revision())
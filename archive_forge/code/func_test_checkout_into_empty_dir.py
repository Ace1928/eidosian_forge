import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
def test_checkout_into_empty_dir(self):
    self.make_controldir('checkout')
    out, err = self.run_bzr(['checkout', 'branch', 'checkout'])
    result = controldir.ControlDir.open('checkout')
    tree = result.open_workingtree()
    branch = result.open_branch()
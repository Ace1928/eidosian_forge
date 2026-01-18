import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
def test_checkout_light_makes_checkout(self):
    self.run_bzr('checkout --lightweight branch checkout')
    source = controldir.ControlDir.open('branch')
    result = controldir.ControlDir.open('checkout')
    self.assertEqual(source.open_branch().controldir.root_transport.base, result.open_branch().controldir.root_transport.base)
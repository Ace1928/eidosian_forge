import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
def test_checkout_dash_r(self):
    out, err = self.run_bzr(['checkout', '-r', '-2', 'branch', 'checkout'])
    result = controldir.ControlDir.open('checkout')
    self.assertEqual([self.rev1], result.open_workingtree().get_parent_ids())
    self.assertPathDoesNotExist('checkout/added_in_2')
import os
from breezy import branch as _mod_branch
from breezy import controldir, errors, workingtree
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport
from breezy.tests.features import HardlinkFeature
def test_checkout_files_from(self):
    branch = _mod_branch.Branch.open('branch')
    self.run_bzr(['checkout', 'branch', 'branch2', '--files-from', 'branch'])
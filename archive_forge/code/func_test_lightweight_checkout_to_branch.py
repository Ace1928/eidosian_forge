from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_lightweight_checkout_to_branch(self):
    reconfiguration, checkout = self.prepare_lightweight_checkout_to_branch()
    reconfiguration.apply()
    checkout_branch = checkout.controldir.open_branch()
    self.assertEqual(checkout_branch.controldir.root_transport.base, checkout.controldir.root_transport.base)
    self.assertEqual(b'rev1', checkout_branch.last_revision())
    repo = checkout.controldir.open_repository()
    repo.get_revision(b'rev1')
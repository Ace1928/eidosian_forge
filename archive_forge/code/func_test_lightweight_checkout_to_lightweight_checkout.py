from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_lightweight_checkout_to_lightweight_checkout(self):
    parent = self.make_branch('parent')
    checkout = parent.create_checkout('checkout', lightweight=True)
    self.assertRaises(reconfigure.AlreadyLightweightCheckout, reconfigure.Reconfigure.to_lightweight_checkout, checkout.controldir)
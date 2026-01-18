from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def prepare_lightweight_checkout_to_branch(self):
    branch = self.make_branch('branch')
    checkout = branch.create_checkout('checkout', lightweight=True)
    checkout.commit('first commit', rev_id=b'rev1')
    reconfiguration = reconfigure.Reconfigure.to_branch(checkout.controldir)
    return (reconfiguration, checkout)
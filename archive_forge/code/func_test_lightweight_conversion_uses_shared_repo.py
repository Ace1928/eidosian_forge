from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_lightweight_conversion_uses_shared_repo(self):
    parent = self.make_branch('parent')
    shared_repo = self.make_repository('repo', shared=True)
    checkout = parent.create_checkout('repo/checkout', lightweight=True)
    reconfigure.Reconfigure.to_tree(checkout.controldir).apply()
    checkout_repo = checkout.controldir.open_branch().repository
    self.assertEqual(shared_repo.controldir.root_transport.base, checkout_repo.controldir.root_transport.base)
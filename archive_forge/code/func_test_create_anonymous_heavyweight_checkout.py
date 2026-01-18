import contextlib
from breezy import branch as _mod_branch
from breezy import config, controldir
from breezy import delta as _mod_delta
from breezy import (errors, lock, merge, osutils, repository, revision, shelf,
from breezy import tree as _mod_tree
from breezy import urlutils
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.tests.http_server import HttpServer
from breezy.transport import memory
def test_create_anonymous_heavyweight_checkout(self):
    """A regular checkout from a readonly branch should succeed."""
    tree_a = self.make_branch_and_tree('a')
    rev_id = tree_a.commit('put some content in the branch')
    url = self.get_readonly_url(osutils.basename(tree_a.branch.base.rstrip('/')))
    t = transport.get_transport_from_url(url)
    if not tree_a.branch.controldir._format.supports_transport(t):
        raise tests.TestNotApplicable('format does not support transport')
    source_branch = _mod_branch.Branch.open(url)
    self.assertRaises((errors.LockError, errors.TransportNotPossible), source_branch.lock_write)
    checkout = source_branch.create_checkout('c')
    self.assertEqual(rev_id, checkout.last_revision())
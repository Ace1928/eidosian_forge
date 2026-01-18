from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_branch_to_lightweight_checkout_fetch_tags(self):
    parent, child, reconfiguration = self.prepare_branch_to_lightweight_checkout()
    child.branch.tags.set_tag('foo', b'bar')
    reconfiguration.apply()
    child = _mod_branch.Branch.open('child')
    self.assertEqual(b'bar', parent.tags.lookup_tag('foo'))
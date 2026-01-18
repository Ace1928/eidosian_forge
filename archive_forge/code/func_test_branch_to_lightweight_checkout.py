from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_branch_to_lightweight_checkout(self):
    parent, child, reconfiguration = self.prepare_branch_to_lightweight_checkout()
    reconfiguration.apply()
    self.assertTrue(reconfiguration._destroy_branch)
    wt = child.controldir.open_workingtree()
    self.assertTrue(parent.repository.has_same_location(wt.branch.repository))
    parent.repository.get_revision(b'new-commit')
    self.assertRaises(errors.NoRepositoryPresent, child.controldir.open_repository)
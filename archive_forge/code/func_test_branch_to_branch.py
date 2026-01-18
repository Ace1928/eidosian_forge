from breezy import branch as _mod_branch
from breezy import (controldir, errors, reconfigure, repository, tests,
from breezy.bzr import branch as _mod_bzrbranch
from breezy.bzr import vf_repository
def test_branch_to_branch(self):
    branch = self.make_branch('branch')
    self.assertRaises(reconfigure.AlreadyBranch, reconfigure.Reconfigure.to_branch, branch.controldir)
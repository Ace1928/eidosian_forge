from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def assertRealBranch(self, b):
    self.assertIsInstance(b, _mod_branch.Branch)
    self.assertFalse(isinstance(b, remote.RemoteBranch))
from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def capture_hook(self, branch):
    self.hook_calls.append(branch)
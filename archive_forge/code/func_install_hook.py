from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def install_hook(self):
    self.hook_calls = []
    _mod_branch.Branch.hooks.install_named_hook('open', self.capture_hook, None)
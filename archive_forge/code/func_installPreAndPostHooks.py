from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def installPreAndPostHooks(self):
    self.pre_hook_calls = self.install_logging_hook('pre')
    self.post_hook_calls = self.install_logging_hook('post')
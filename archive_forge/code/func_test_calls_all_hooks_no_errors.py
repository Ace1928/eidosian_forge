from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def test_calls_all_hooks_no_errors(self):
    """If multiple hooks are registered, all are called (if none raise
        errors).
        """
    branch = self.make_branch('source')
    hook_calls_1 = self.install_logging_hook('post')
    hook_calls_2 = self.install_logging_hook('post')
    self.assertIsNot(hook_calls_1, hook_calls_2)
    branch.set_last_revision_info(0, revision.NULL_REVISION)
    if isinstance(branch, remote.RemoteBranch):
        count = 2
    else:
        count = 1
    self.assertEqual(len(hook_calls_1), count)
    self.assertEqual(len(hook_calls_2), count)
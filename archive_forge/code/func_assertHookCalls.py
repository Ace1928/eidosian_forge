from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def assertHookCalls(self, expected_params, branch, hook_calls=None, pre=False):
    if hook_calls is None:
        hook_calls = self.hook_calls
    if isinstance(branch, remote.RemoteBranch):
        if pre:
            offset = 0
        else:
            offset = 1
        self.assertEqual(expected_params, hook_calls[offset])
        self.assertEqual(2, len(hook_calls))
    else:
        self.assertEqual([expected_params], hook_calls)
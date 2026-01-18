from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def test_empty_history(self):
    branch = self.make_branch('source')
    hook_calls = self.install_logging_hook('post')
    branch.set_last_revision_info(0, revision.NULL_REVISION)
    expected_params = _mod_branch.ChangeBranchTipParams(branch, 0, 0, revision.NULL_REVISION, revision.NULL_REVISION)
    self.assertHookCalls(expected_params, branch, hook_calls)
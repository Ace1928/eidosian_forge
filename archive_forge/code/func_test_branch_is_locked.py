from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def test_branch_is_locked(self):
    """The branch passed to the hook is locked."""
    branch = self.make_branch('source')

    def assertBranchIsLocked(params):
        self.assertTrue(params.branch.is_locked())
    _mod_branch.Branch.hooks.install_named_hook('post_change_branch_tip', assertBranchIsLocked, None)
    branch.set_last_revision_info(0, revision.NULL_REVISION)
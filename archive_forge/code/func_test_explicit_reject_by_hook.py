from breezy import branch as _mod_branch
from breezy import errors, revision, tests
from breezy.bzr import remote
from breezy.tests import test_server
def test_explicit_reject_by_hook(self):
    """If a hook raises TipChangeRejected, the change does not take effect.

        TipChangeRejected exceptions are propagated, not wrapped in HookFailed.
        """
    branch = self.make_branch_with_revision_ids(b'one-\xc2\xb5', b'two-\xc2\xb5')

    def hook_that_rejects(params):
        raise errors.TipChangeRejected('rejection message')
    _mod_branch.Branch.hooks.install_named_hook('pre_change_branch_tip', hook_that_rejects, None)
    self.assertRaises(errors.TipChangeRejected, branch.set_last_revision_info, 0, revision.NULL_REVISION)
    self.assertEqual((2, b'two-\xc2\xb5'), branch.last_revision_info())
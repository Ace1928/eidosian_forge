from breezy import branch, errors, tests
from breezy.bzr import remote
from breezy.tests import per_branch
from breezy.transport import FileExists, NoSuchFile
def test_create_clone_on_transport_stacked_hooks_get_stacked_branch(self):
    tree = self.make_branch_and_tree('source')
    tree.commit('a commit')
    trunk = tree.branch.create_clone_on_transport(self.get_transport('trunk'))
    revid = tree.commit('a second commit')
    target_transport = self.get_transport('target')
    self.hook_calls = []
    branch.Branch.hooks.install_named_hook('pre_change_branch_tip', self.assertBranchHookBranchIsStacked, None)
    try:
        result = tree.branch.create_clone_on_transport(target_transport, stacked_on=trunk.base)
    except branch.UnstackableBranchFormat:
        if not trunk.repository._format.supports_full_versioned_files:
            raise tests.TestNotApplicable('can not stack on format')
        raise
    self.assertEqual(revid, result.last_revision())
    self.assertEqual(trunk.base, result.get_stacked_on_url())
    if isinstance(result, remote.RemoteBranch):
        expected_calls = 2
    else:
        expected_calls = 1
    self.assertEqual(expected_calls, len(self.hook_calls))
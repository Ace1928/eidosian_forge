from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_hooks_with_no_hooks(self):
    self.make_branch('.')
    out, err = self.run_bzr('hooks')
    self.assertEqual(err, '')
    for hook_type in Branch.hooks:
        self._check_hooks_output(out, {})
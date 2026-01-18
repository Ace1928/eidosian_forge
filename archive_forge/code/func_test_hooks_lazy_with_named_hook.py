from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_hooks_lazy_with_named_hook(self):
    self.make_branch('.')

    def foo():
        return
    Branch.hooks.install_named_hook_lazy('post_push', 'breezy.tests.blackbox.test_hooks', '_foo_hook', 'hook has a name')
    out, err = self.run_bzr('hooks')
    self._check_hooks_output(out, {'post_push': ['hook has a name']})
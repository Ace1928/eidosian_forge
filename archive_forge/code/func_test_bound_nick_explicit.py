import breezy
from breezy import branch, osutils, tests
def test_bound_nick_explicit(self):
    """Bind should update explicit nick."""
    base = self.make_branch_and_tree('base')
    child = self.make_branch_and_tree('child')
    self.run_bzr('nick explicit_nick', working_dir='child')
    self.assertNick('explicit_nick', working_dir='child', explicit=True)
    self.run_bzr('bind ../base', working_dir='child')
    self.assertNick(base.branch.nick, working_dir='child', explicit=True)
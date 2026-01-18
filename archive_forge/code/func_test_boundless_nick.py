import breezy
from breezy import branch, osutils, tests
def test_boundless_nick(self):
    """Nick defaults to implicit local nick when bound branch is AWOL"""
    base = self.make_branch_and_tree('base')
    child = self.make_branch_and_tree('child')
    self.run_bzr('bind ../base', working_dir='child')
    self.assertNick(base.branch.nick, working_dir='child', explicit=False)
    osutils.rmtree('base')
    self.assertNick('child', working_dir='child', explicit=False)
import breezy
from breezy import branch, osutils, tests
def test_nick_command(self):
    """brz nick for viewing, setting nicknames"""
    self.make_branch_and_tree('me.dev')
    self.assertNick('me.dev', working_dir='me.dev')
    self.run_bzr('nick moo', working_dir='me.dev')
    self.assertNick('moo', working_dir='me.dev')
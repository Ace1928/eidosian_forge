from .. import TestCaseWithTransport
def test_fetch_ghosts_with_saved(self):
    wt = self.make_branch_and_tree('.')
    wt.branch.set_parent('.')
    self.run_bzr('fetch-ghosts')
from ..fetch_ghosts import GhostFetcher
from . import TestCaseWithTransport
def test_fetch_ghosts_failure(self):
    tree = self.prepare_with_ghosts()
    branch = self.make_branch('branch')
    GhostFetcher(tree.branch, branch).run()
    self.assertFalse(tree.branch.repository.has_revision(b'ghost-id'))
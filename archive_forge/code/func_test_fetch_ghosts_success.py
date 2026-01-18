from ..fetch_ghosts import GhostFetcher
from . import TestCaseWithTransport
def test_fetch_ghosts_success(self):
    tree = self.prepare_with_ghosts()
    ghost_tree = self.make_branch_and_tree('ghost_tree')
    ghost_tree.commit('ghost', rev_id=b'ghost-id')
    GhostFetcher(tree.branch, ghost_tree.branch).run()
    self.assertTrue(tree.branch.repository.has_revision(b'ghost-id'))
from breezy import errors
from breezy.tests.per_repository_reference import \
def test_known_pair_is_after(self):
    repo = self.tree.branch.repository
    repo.lock_read()
    self.addCleanup(repo.unlock)
    self.assertRaises(errors.RevnoOutOfBounds, repo.get_rev_id_for_revno, 3, (2, self.revid2))
from breezy.tests.per_repository import TestCaseWithRepository
def test_gather_stats(self):
    """First smoke test covering the refactoring into the Repository api."""
    tree = self.make_branch_and_memory_tree('.')
    tree.lock_write()
    tree.add('')
    rev1 = tree.commit('first post', committer='person 1', timestamp=1170491381, timezone=0)
    rev2 = tree.commit('second post', committer='person 2', timestamp=1171491381, timezone=0)
    rev3 = tree.commit('third post', committer='person 3', timestamp=1172491381, timezone=0)
    tree.unlock()
    stats = tree.branch.repository.gather_stats(rev2, committers=False)
    self.assertEqual(stats['firstrev'], (1170491381.0, 0))
    self.assertEqual(stats['latestrev'], (1171491381.0, 0))
    self.assertEqual(stats['revisions'], 3)
    stats = tree.branch.repository.gather_stats(rev2, committers=True)
    self.assertEqual(2, stats['committers'])
    self.assertEqual((1170491381.0, 0), stats['firstrev'])
    self.assertEqual((1171491381.0, 0), stats['latestrev'])
    self.assertEqual(3, stats['revisions'])
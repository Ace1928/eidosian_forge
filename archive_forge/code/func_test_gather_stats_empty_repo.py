from breezy.tests.per_repository import TestCaseWithRepository
def test_gather_stats_empty_repo(self):
    """An empty repository still has revisions."""
    tree = self.make_branch_and_memory_tree('.')
    stats = tree.branch.repository.gather_stats()
    self.assertEqual(0, stats['revisions'])
    self.assertFalse('committers' in stats)
    self.assertFalse('firstrev' in stats)
    self.assertFalse('latestrev' in stats)
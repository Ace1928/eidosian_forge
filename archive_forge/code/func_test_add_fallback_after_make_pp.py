from breezy.tests.per_repository import TestCaseWithRepository
def test_add_fallback_after_make_pp(self):
    """Fallbacks added after _make_parents_provider are used by that
        provider.
        """
    referring_repo = self.make_repository('repo')
    pp = referring_repo._make_parents_provider()
    self.addCleanup(referring_repo.lock_read().unlock)
    self.assertEqual({}, pp.get_parent_map([b'revid2']))
    wt_a = self.make_branch_and_tree('fallback')
    wt_a.commit('first commit', rev_id=b'revid1')
    wt_a.commit('second commit', rev_id=b'revid2')
    fallback_repo = wt_a.branch.repository
    referring_repo.add_fallback_repository(fallback_repo)
    self.assertEqual((b'revid1',), pp.get_parent_map([b'revid2'])[b'revid2'])
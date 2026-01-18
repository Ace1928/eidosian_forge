import os
from .. import check, osutils
from ..commit import PointlessCommit
from . import TestCaseWithTransport
from .features import SymlinkFeature
from .matchers import RevisionHistoryMatches
def test_merge_commit_empty(self):
    """Simple commit of two-way merge of empty trees."""
    wtx = self.make_branch_and_tree('x')
    base_rev = wtx.commit('common parent')
    bx = wtx.branch
    wty = wtx.controldir.sprout('y').open_workingtree()
    by = wty.branch
    wtx.commit('commit one', rev_id=b'x@u-0-1', allow_pointless=True)
    wty.commit('commit two', rev_id=b'y@u-0-1', allow_pointless=True)
    by.fetch(bx)
    self.assertRaises(PointlessCommit, wty.commit, 'no changes yet', rev_id=b'y@u-0-2', allow_pointless=False)
    wty.merge_from_branch(bx)
    wty.commit('merge from x', rev_id=b'y@u-0-2', allow_pointless=False)
    self.assertEqual(by.revno(), 3)
    graph = wty.branch.repository.get_graph()
    self.addCleanup(wty.lock_read().unlock)
    self.assertThat(by, RevisionHistoryMatches([base_rev, b'y@u-0-1', b'y@u-0-2']))
    rev = by.repository.get_revision(b'y@u-0-2')
    self.assertEqual(rev.parent_ids, [b'y@u-0-1', b'x@u-0-1'])
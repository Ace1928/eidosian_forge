import os
from .. import check, osutils
from ..commit import PointlessCommit
from . import TestCaseWithTransport
from .features import SymlinkFeature
from .matchers import RevisionHistoryMatches
def test_merge_new_file(self):
    """Commit merge of two trees with no overlapping files."""
    wtx = self.make_branch_and_tree('x')
    base_rev = wtx.commit('common parent')
    bx = wtx.branch
    wtx.commit('establish root id')
    wty = wtx.controldir.sprout('y').open_workingtree()
    self.assertEqual(wtx.path2id(''), wty.path2id(''))
    by = wty.branch
    self.build_tree(['x/ecks', 'y/why'])
    wtx.add(['ecks'], ids=[b'ecks-id'])
    wty.add(['why'], ids=[b'why-id'])
    wtx.commit('commit one', rev_id=b'x@u-0-1', allow_pointless=True)
    wty.commit('commit two', rev_id=b'y@u-0-1', allow_pointless=True)
    wty.merge_from_branch(bx)
    self.assertRaises(Exception, wty.commit, 'partial commit', allow_pointless=False, specific_files=['ecks'])
    wty.commit('merge from x', rev_id=b'y@u-0-2', allow_pointless=False)
    tree = by.repository.revision_tree(b'y@u-0-2')
    self.assertEqual(tree.get_file_revision('ecks'), b'x@u-0-1')
    self.assertEqual(tree.get_file_revision('why'), b'y@u-0-1')
    check.check_dwim(bx.base, False, True, True)
    check.check_dwim(by.base, False, True, True)
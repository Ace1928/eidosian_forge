from breezy.errors import FetchLimitUnsupported, NoRoundtrippingSupport
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_fetch_revisions(self):
    """Test fetch-revision operation."""
    wt = self.make_from_branch_and_tree('b1')
    b1 = wt.branch
    self.build_tree_contents([('b1/foo', b'hello')])
    wt.add(['foo'])
    rev1 = wt.commit('lala!', allow_pointless=False)
    b2 = self.make_to_branch('b2')
    try:
        b2.fetch(b1)
    except NoRoundtrippingSupport:
        raise TestNotApplicable('lossless cross-vcs fetch %r to %r not supported' % (b1, b2))
    self.assertEqual(NULL_REVISION, b2.last_revision())
    b2.repository.get_revision(rev1)
    tree = b2.repository.revision_tree(rev1)
    tree.lock_read()
    self.addCleanup(tree.unlock)
    self.assertEqual(tree.get_file_text('foo'), b'hello')
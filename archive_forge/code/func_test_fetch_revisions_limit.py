from breezy.errors import FetchLimitUnsupported, NoRoundtrippingSupport
from breezy.revision import NULL_REVISION
from breezy.tests import TestNotApplicable
from breezy.tests.per_interbranch import TestCaseWithInterBranch
def test_fetch_revisions_limit(self):
    """Test fetch-revision operation."""
    builder = self.make_branch_builder('b1', format=self.branch_format_from._matchingcontroldir)
    builder.start_series()
    rev1 = builder.build_commit()
    rev2 = builder.build_commit()
    rev3 = builder.build_commit()
    builder.finish_series()
    b1 = builder.get_branch()
    b2 = self.make_to_branch('b2')
    try:
        b2.fetch(b1, limit=1)
    except FetchLimitUnsupported:
        raise TestNotApplicable('interbranch does not support fetch limits')
    except NoRoundtrippingSupport:
        raise TestNotApplicable('lossless cross-vcs fetch %r to %r not supported' % (b1, b2))
    self.assertEqual(NULL_REVISION, b2.last_revision())
    self.assertEqual({rev1}, b2.repository.has_revisions([rev1, rev2, rev3]))
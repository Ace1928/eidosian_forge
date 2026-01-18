import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def test_get_apparent_authors_no_committer(self):
    r = revision.Revision('1')
    self.assertEqual([], r.get_apparent_authors())
import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def test_get_apparent_authors(self):
    r = revision.Revision('1')
    r.committer = 'A'
    self.assertEqual(['A'], r.get_apparent_authors())
    r.properties['author'] = 'B'
    self.assertEqual(['B'], r.get_apparent_authors())
    r.properties['authors'] = 'C\nD'
    self.assertEqual(['C', 'D'], r.get_apparent_authors())
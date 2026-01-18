import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def test_no_bugs(self):
    r = revision.Revision('1')
    self.assertEqual([], list(r.iter_bugs()))
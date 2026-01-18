import warnings
from breezy import bugtracker, revision
from breezy.revision import NULL_REVISION
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tests.matchers import MatchesAncestry
def test_invalid_status(self):
    r = revision.Revision('1', properties={'bugs': 'http://example.com/bugs/1 faxed'})
    self.assertRaises(bugtracker.InvalidBugStatus, list, r.iter_bugs())
from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_check_bug_id_doesnt_accept_non_project_integers(self):
    """Rejects non-integers as bug IDs."""
    tracker = bugtracker.ProjectIntegerBugTracker('xxx', 'http://bugs.example.com/{project}/{id}')
    self.assertRaises(bugtracker.MalformedBugIdentifier, tracker.check_bug_id, 'red')
    self.assertRaises(bugtracker.MalformedBugIdentifier, tracker.check_bug_id, '1234')
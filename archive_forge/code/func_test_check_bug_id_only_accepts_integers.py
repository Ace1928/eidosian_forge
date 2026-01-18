from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_check_bug_id_only_accepts_integers(self):
    """A UniqueIntegerBugTracker accepts integers as bug IDs."""
    tracker = bugtracker.UniqueIntegerBugTracker('xxx', 'http://bugs.example.com/')
    tracker.check_bug_id('1234')
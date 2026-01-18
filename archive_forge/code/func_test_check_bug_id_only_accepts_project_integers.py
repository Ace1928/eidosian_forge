from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_check_bug_id_only_accepts_project_integers(self):
    """Accepts integers as bug IDs."""
    tracker = bugtracker.ProjectIntegerBugTracker('xxx', 'http://bugs.example.com/{project}/{id}')
    tracker.check_bug_id('project/1234')
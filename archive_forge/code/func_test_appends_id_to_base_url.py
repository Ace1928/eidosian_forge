from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_appends_id_to_base_url(self):
    """The URL of a bug is the base URL joined to the identifier."""
    tracker = bugtracker.ProjectIntegerBugTracker('xxx', 'http://bugs.example.com/{project}/{id}')
    self.assertEqual('http://bugs.example.com/foo/1234', tracker.get_bug_url('foo/1234'))
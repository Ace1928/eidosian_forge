from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_github(self):
    branch = self.make_branch('some_branch')
    tracker = bugtracker.tracker_registry.get_tracker('github', branch)
    self.assertEqual('https://github.com/breezy-team/breezy/issues/1234', tracker.get_bug_url('breezy-team/breezy/1234'))
from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_gnome_registered(self):
    branch = self.make_branch('some_branch')
    tracker = bugtracker.tracker_registry.get_tracker('gnome', branch)
    self.assertEqual('http://bugzilla.gnome.org/show_bug.cgi?id=1234', tracker.get_bug_url('1234'))
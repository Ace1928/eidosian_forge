from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_generic_registered_non_integer(self):
    branch = self.make_branch('some_branch')
    config = branch.get_config()
    config.set_user_option('bugtracker_foo_url', 'http://bugs.example.com/{id}/view.html')
    tracker = bugtracker.tracker_registry.get_tracker('foo', branch)
    self.assertEqual('http://bugs.example.com/ABC-1234/view.html', tracker.get_bug_url('ABC-1234'))
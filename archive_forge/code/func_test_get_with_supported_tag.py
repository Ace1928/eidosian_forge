from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_get_with_supported_tag(self):
    """If asked for a valid tag, return a tracker instance that can map bug
        IDs to <base_url>/<bug_area> + <bug_id>."""
    bugtracker.tracker_registry.register('some', self.tracker)
    self.addCleanup(bugtracker.tracker_registry.remove, 'some')
    branch = self.make_branch('some_branch')
    config = branch.get_config()
    config.set_user_option('some_twisted_url', self.url)
    tracker = self.tracker.get('twisted', branch)
    self.assertEqual(urlutils.join(self.url, 'ticket/') + '1234', tracker.get_bug_url('1234'))
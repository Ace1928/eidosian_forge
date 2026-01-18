from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_launchpad_registered(self):
    """The Launchpad bug tracker should be registered by default and
        generate Launchpad bug page URLs.
        """
    branch = self.make_branch('some_branch')
    tracker = bugtracker.tracker_registry.get_tracker('lp', branch)
    self.assertEqual('https://launchpad.net/bugs/1234', tracker.get_bug_url('1234'))
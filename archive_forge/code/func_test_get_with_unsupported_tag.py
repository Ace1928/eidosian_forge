from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_get_with_unsupported_tag(self):
    """If asked for an unrecognized or unconfigured tag, return None."""
    branch = self.make_branch('some_branch')
    self.assertEqual(None, self.tracker.get('lp', branch))
    self.assertEqual(None, self.tracker.get('twisted', branch))
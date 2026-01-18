from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_decoding_invalid(self):
    self.assertRaises(bugtracker.InvalidLineInBugsProperty, list, bugtracker.decode_bug_urls('http://example.com/bugs/ 1 fixed\n'))
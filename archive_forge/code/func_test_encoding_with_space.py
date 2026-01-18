from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_encoding_with_space(self):
    self.assertRaises(bugtracker.InvalidBugUrl, bugtracker.encode_fixes_bug_urls, [('http://example.com/bugs/ 1', 'fixed')])
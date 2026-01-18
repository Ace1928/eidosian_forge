from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_decoding_one(self):
    self.assertEqual([('http://example.com/bugs/1', 'fixed')], list(bugtracker.decode_bug_urls('http://example.com/bugs/1 fixed')))
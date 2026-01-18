from .. import bugtracker, urlutils
from . import TestCase, TestCaseWithMemoryTransport
def test_encoding_one(self):
    self.assertEqual('http://example.com/bugs/1 fixed', bugtracker.encode_fixes_bug_urls([('http://example.com/bugs/1', 'fixed')]))
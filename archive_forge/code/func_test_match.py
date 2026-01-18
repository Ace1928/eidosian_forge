from testtools.matchers import *
from ...tests import CapturedCall, TestCase
from ..smart.client import CallHookParams
from .matchers import *
def test_match(self):
    calls = [self._make_call(b'append', [b'file']), self._make_call(b'Branch.get_config_file', [])]
    mismatch = ContainsNoVfsCalls().match(calls)
    self.assertIsNot(None, mismatch)
    self.assertEqual([calls[0].call], mismatch.vfs_calls)
    self.assertIn(mismatch.describe(), ["no VFS calls expected, got: b'append'(b'file')", "no VFS calls expected, got: append('file')"])
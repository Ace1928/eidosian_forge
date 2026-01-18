from testtools.matchers import *
from ...tests import CapturedCall, TestCase
from ..smart.client import CallHookParams
from .matchers import *
def test_ignores_unknown(self):
    calls = [self._make_call('unknown', [])]
    self.assertIs(None, ContainsNoVfsCalls().match(calls))
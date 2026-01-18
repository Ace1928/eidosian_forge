import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_call_any(self):
    self.assertEqual(call, ANY)
    m = MagicMock()
    int(m)
    self.assertEqual(m.mock_calls, [ANY])
    self.assertEqual([ANY], m.mock_calls)
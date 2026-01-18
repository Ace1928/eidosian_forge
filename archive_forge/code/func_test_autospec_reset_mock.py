import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_autospec_reset_mock(self):
    m = create_autospec(int)
    int(m)
    m.reset_mock()
    self.assertEqual(m.__int__.call_count, 0)
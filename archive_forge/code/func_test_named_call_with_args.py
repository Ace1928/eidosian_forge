import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_named_call_with_args(self):
    args = _Call(('foo', (1, 2, 3), {}))
    self.assertEqual(args, ('foo', (1, 2, 3)))
    self.assertEqual(args, ('foo', (1, 2, 3), {}))
    self.assertNotEqual(args, ((1, 2, 3),))
    self.assertNotEqual(args, ((1, 2, 3), {}))
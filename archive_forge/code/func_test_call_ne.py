import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_call_ne(self):
    self.assertNotEqual(_Call(((1, 2, 3),)), call(1, 2))
    self.assertFalse(_Call(((1, 2, 3),)) != call(1, 2, 3))
    self.assertTrue(_Call(((1, 2), {})) != call(1, 2, 3))
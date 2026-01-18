import six
import unittest2 as unittest
from mock import (
from mock.mock import _Call, _CallList
from datetime import datetime
def test_method_calls(self):

    class Sub(SomeClass):
        attr = SomeClass()
    mock = create_autospec(Sub)
    mock.one(1, 2)
    mock.two()
    mock.three(3)
    expected = [call.one(1, 2), call.two(), call.three(3)]
    self.assertEqual(mock.method_calls, expected)
    mock.attr.one(1, 2)
    mock.attr.two()
    mock.attr.three(3)
    expected.extend([call.attr.one(1, 2), call.attr.two(), call.attr.three(3)])
    self.assertEqual(mock.method_calls, expected)
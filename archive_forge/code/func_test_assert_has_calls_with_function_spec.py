import copy
import pickle
import sys
import tempfile
import six
import unittest2 as unittest
import mock
from mock import (
from mock.mock import _CallList
from mock.tests.support import (
def test_assert_has_calls_with_function_spec(self):

    def f(a, b, c, d=None):
        pass
    mock = Mock(spec=f)
    mock(1, b=2, c=3)
    mock(4, 5, c=6, d=7)
    mock(10, 11, c=12)
    calls = [('', (1, 2, 3), {}), ('', (4, 5, 6), {'d': 7}), ((10, 11, 12), {})]
    mock.assert_has_calls(calls)
    mock.assert_has_calls(calls, any_order=True)
    mock.assert_has_calls(calls[1:])
    mock.assert_has_calls(calls[1:], any_order=True)
    mock.assert_has_calls(calls[:-1])
    mock.assert_has_calls(calls[:-1], any_order=True)
    calls = list(reversed(calls))
    with self.assertRaises(AssertionError):
        mock.assert_has_calls(calls)
    mock.assert_has_calls(calls, any_order=True)
    with self.assertRaises(AssertionError):
        mock.assert_has_calls(calls[1:])
    mock.assert_has_calls(calls[1:], any_order=True)
    with self.assertRaises(AssertionError):
        mock.assert_has_calls(calls[:-1])
    mock.assert_has_calls(calls[:-1], any_order=True)
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
def test_assert_has_calls_any_order(self):
    mock = Mock()
    mock(1, 2)
    mock(a=3)
    mock(3, 4)
    mock(b=6)
    mock(b=6)
    kalls = [call(1, 2), ({'a': 3},), ((3, 4),), ((), {'a': 3}), ('', (1, 2)), ('', {'a': 3}), ('', (1, 2), {}), ('', (), {'a': 3})]
    for kall in kalls:
        mock.assert_has_calls([kall], any_order=True)
    for kall in (call(1, '2'), call(b=3), call(), 3, None, 'foo'):
        self.assertRaises(AssertionError, mock.assert_has_calls, [kall], any_order=True)
    kall_lists = [[call(1, 2), call(b=6)], [call(3, 4), call(1, 2)], [call(b=6), call(b=6)]]
    for kall_list in kall_lists:
        mock.assert_has_calls(kall_list, any_order=True)
    kall_lists = [[call(b=6), call(b=6), call(b=6)], [call(1, 2), call(1, 2)], [call(3, 4), call(1, 2), call(5, 7)], [call(b=6), call(3, 4), call(b=6), call(1, 2), call(b=6)]]
    for kall_list in kall_lists:
        self.assertRaises(AssertionError, mock.assert_has_calls, kall_list, any_order=True)
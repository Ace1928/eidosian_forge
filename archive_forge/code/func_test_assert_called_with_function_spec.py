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
def test_assert_called_with_function_spec(self):

    def f(a, b, c, d=None):
        pass
    mock = Mock(spec=f)
    mock(1, b=2, c=3)
    mock.assert_called_with(1, 2, 3)
    mock.assert_called_with(a=1, b=2, c=3)
    self.assertRaises(AssertionError, mock.assert_called_with, 1, b=3, c=2)
    with self.assertRaises(AssertionError) as cm:
        mock.assert_called_with(e=8)
    if hasattr(cm.exception, '__cause__'):
        self.assertIsInstance(cm.exception.__cause__, TypeError)
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
def test_assert_called_with_method_spec(self):

    def _check(mock):
        mock(1, b=2, c=3)
        mock.assert_called_with(1, 2, 3)
        mock.assert_called_with(a=1, b=2, c=3)
        self.assertRaises(AssertionError, mock.assert_called_with, 1, b=3, c=2)
    mock = Mock(spec=Something().meth)
    _check(mock)
    mock = Mock(spec=Something.cmeth)
    _check(mock)
    mock = Mock(spec=Something().cmeth)
    _check(mock)
    mock = Mock(spec=Something.smeth)
    _check(mock)
    mock = Mock(spec=Something().smeth)
    _check(mock)
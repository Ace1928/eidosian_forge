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
def test_assert_not_called(self):
    m = Mock()
    m.hello.assert_not_called()
    m.hello()
    with self.assertRaises(AssertionError):
        m.hello.assert_not_called()
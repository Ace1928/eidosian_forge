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
def test_assert_called_once_with_message(self):
    mock = Mock(name='geoffrey')
    self.assertRaisesRegex(AssertionError, "Expected 'geoffrey' to be called once\\.", mock.assert_called_once_with)
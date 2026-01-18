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
def test_mock_open_alter_readline(self):
    mopen = mock.mock_open(read_data='foo\nbarn')
    mopen.return_value.readline.side_effect = lambda *args: 'abc'
    first = mopen().readline()
    second = mopen().readline()
    self.assertEqual('abc', first)
    self.assertEqual('abc', second)
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
def assert_attrs(mock):
    names = ('call_args_list', 'method_calls', 'mock_calls')
    for name in names:
        attr = getattr(mock, name)
        self.assertIsInstance(attr, _CallList)
        self.assertIsInstance(attr, list)
        self.assertEqual(attr, [])
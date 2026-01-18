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
def test_call_args_comparison(self):
    mock = Mock()
    mock()
    mock(sentinel.Arg)
    mock(kw=sentinel.Kwarg)
    mock(sentinel.Arg, kw=sentinel.Kwarg)
    self.assertEqual(mock.call_args_list, [(), ((sentinel.Arg,),), ({'kw': sentinel.Kwarg},), ((sentinel.Arg,), {'kw': sentinel.Kwarg})])
    self.assertEqual(mock.call_args, ((sentinel.Arg,), {'kw': sentinel.Kwarg}))
    self.assertFalse(mock.call_args == 'a long sequence')
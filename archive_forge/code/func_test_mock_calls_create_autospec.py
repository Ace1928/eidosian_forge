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
def test_mock_calls_create_autospec(self):

    def f(a, b):
        pass
    obj = Iter()
    obj.f = f
    funcs = [create_autospec(f), create_autospec(obj).f]
    for func in funcs:
        func(1, 2)
        func(3, 4)
        self.assertEqual(func.mock_calls, [call(1, 2), call(3, 4)])
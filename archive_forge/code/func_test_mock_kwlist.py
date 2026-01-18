import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_mock_kwlist(self):
    with mock.patch.object(keyword, 'kwlist', new=['abcd']):
        self.assertEqual(self.com.matches(3, 'abc', locals_={}), None)
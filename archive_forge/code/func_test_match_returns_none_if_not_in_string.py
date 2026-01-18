import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
@mock.patch(glob_function, new=lambda text: [])
def test_match_returns_none_if_not_in_string(self):
    self.assertEqual(self.completer.matches(2, 'abcd'), None)
import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_first_non_none_completer_matches_are_returned(self):
    a = completer([])
    b = completer(['a'])
    self.assertEqual(autocomplete.get_completer([a, b], 0, ''), ([], None))
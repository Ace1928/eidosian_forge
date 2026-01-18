import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_first_completer_returns_None(self):
    a = completer(None)
    b = completer(['a'])
    self.assertEqual(autocomplete.get_completer([a, b], 0, ''), (['a'], b))
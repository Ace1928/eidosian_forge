import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_completions_starting_with_different_letters(self):
    matches = self.matches_from_completions(2, ' a', 'class Foo:\n a', ['adsf'], [Completion('Abc', 'bc'), Completion('Cbc', 'bc')])
    self.assertEqual(matches, None)
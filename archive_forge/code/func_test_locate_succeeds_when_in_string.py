import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_locate_succeeds_when_in_string(self):
    self.assertEqual(self.completer.locate(4, "a'bc'd"), LinePart(2, 4, 'bc'))
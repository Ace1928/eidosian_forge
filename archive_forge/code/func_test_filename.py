import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_filename(self):
    completer = autocomplete.FilenameCompletion()
    last_part_of_filename = completer.format
    self.assertEqual(last_part_of_filename('abc'), 'abc')
    self.assertEqual(last_part_of_filename('abc/'), 'abc/')
    self.assertEqual(last_part_of_filename('abc/efg'), 'efg')
    self.assertEqual(last_part_of_filename('abc/efg/'), 'efg/')
    self.assertEqual(last_part_of_filename('/abc'), 'abc')
    self.assertEqual(last_part_of_filename('ab.c/e.f.g/'), 'e.f.g/')
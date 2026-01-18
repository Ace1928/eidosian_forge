import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
@mock.patch('os.path.sep', new='/')
def test_formatting_takes_just_last_part(self):
    self.assertEqual(self.completer.format('/hello/there/'), 'there/')
    self.assertEqual(self.completer.format('/hello/there'), 'there')
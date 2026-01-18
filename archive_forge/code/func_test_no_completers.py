import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_no_completers(self):
    self.assertTupleEqual(autocomplete.get_completer([], 0, ''), ([], None))
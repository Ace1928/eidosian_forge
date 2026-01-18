import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_issue_491(self):
    self.assertNotEqual(self.completer.matches(9, '"a[a.l-1]'), None)
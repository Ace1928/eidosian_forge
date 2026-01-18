import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_completions_are_unicode(self):
    for m in self.com.matches(1, 'a', locals_={'abc': 10}):
        self.assertIsInstance(m, str)
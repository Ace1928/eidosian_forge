import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_dictionaries_complete(self):
    self.assertSetEqual(self.com.matches(7, 'a["b"].', locals_={'a': {'b': Foo()}}), {'method', 'a', 'b'})
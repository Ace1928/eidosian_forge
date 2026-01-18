import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
def test_descriptor_attributes_not_run(self):
    com = autocomplete.AttrCompletion()
    self.assertSetEqual(com.matches(2, 'a.', locals_={'a': Properties()}), {'a.b', 'a.a', 'a.method', 'a.asserts_when_called'})
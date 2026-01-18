import inspect
import keyword
import unittest
from collections import namedtuple
from unittest import mock
from bpython import autocomplete, inspection
from bpython.line import LinePart
@unittest.skip('TODO, subclasses do not complete yet')
def test_list_subclasses_complete(self):

    class ListSubclass(list):
        pass
    self.assertSetEqual(self.com.matches(5, 'a[0].', locals_={'a': ListSubclass([Foo()])}), {'method', 'a', 'b'})
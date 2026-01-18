import ast
import numbers
import sys
import unittest
from bpython.simpleeval import (
def test_attribute_access(self):

    class Foo:
        abc = 1
    self.assertEqual(simple_eval('foo.abc', {'foo': Foo()}), 1)
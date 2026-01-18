from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import unittest
from pasta.base import test_utils
def test_different_number_of_nodes(self):
    src1 = 'X = 1\ndef Foo():\n  return None\n'
    src2 = src1 + 'Y = 2\n'
    t1 = ast.parse(src1)
    t2 = ast.parse(src2)
    with self.assertRaises(AssertionError):
        self.checkAstsEqual(t1, t2)
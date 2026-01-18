import collections
import doctest
import types
from typing import Any, Iterator, Mapping
import unittest
from absl.testing import parameterized
import attr
import numpy as np
import tree
import wrapt
def testIsSequence(self):
    self.assertFalse(tree.is_nested('1234'))
    self.assertFalse(tree.is_nested(b'1234'))
    self.assertFalse(tree.is_nested(u'1234'))
    self.assertFalse(tree.is_nested(bytearray('1234', 'ascii')))
    self.assertTrue(tree.is_nested([1, 3, [4, 5]]))
    self.assertTrue(tree.is_nested(((7, 8), (5, 6))))
    self.assertTrue(tree.is_nested([]))
    self.assertTrue(tree.is_nested({'a': 1, 'b': 2}))
    self.assertFalse(tree.is_nested(set([1, 2])))
    ones = np.ones([2, 3])
    self.assertFalse(tree.is_nested(ones))
    self.assertFalse(tree.is_nested(np.tanh(ones)))
    self.assertFalse(tree.is_nested(np.ones((4, 5))))
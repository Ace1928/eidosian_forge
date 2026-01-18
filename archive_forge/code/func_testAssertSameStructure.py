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
def testAssertSameStructure(self):
    tree.assert_same_structure(STRUCTURE1, STRUCTURE2)
    tree.assert_same_structure('abc', 1.0)
    tree.assert_same_structure(b'abc', 1.0)
    tree.assert_same_structure(u'abc', 1.0)
    tree.assert_same_structure(bytearray('abc', 'ascii'), 1.0)
    tree.assert_same_structure('abc', np.array([0, 1]))
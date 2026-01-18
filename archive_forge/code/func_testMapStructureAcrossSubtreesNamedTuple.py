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
def testMapStructureAcrossSubtreesNamedTuple(self):
    Foo = collections.namedtuple('Foo', ['x', 'y'])
    Bar = collections.namedtuple('Bar', ['x'])
    shallow = Bar(1)
    deep1 = Foo(1, (1, 0))
    deep2 = Foo(2, (2, 0))
    summed = tree.map_structure_up_to(shallow, lambda *args: sum(args), deep1, deep2)
    expected = Bar(3)
    self.assertEqual(summed, expected)
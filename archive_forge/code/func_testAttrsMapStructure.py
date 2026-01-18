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
@parameterized.parameters([(1, 2, 3), ({'B': 10, 'A': 20}, [1, 2], 3), ((1, 2), [3, 4], 5), (collections.namedtuple('Point', ['x', 'y'])(1, 2), 3, 4), wrapt.ObjectProxy((collections.namedtuple('Point', ['x', 'y'])(1, 2), 3, 4))])
def testAttrsMapStructure(self, *field_values):

    @attr.s
    class SampleAttr(object):
        field3 = attr.ib()
        field1 = attr.ib()
        field2 = attr.ib()
    structure = SampleAttr(*field_values)
    new_structure = tree.map_structure(lambda x: x, structure)
    self.assertEqual(structure, new_structure)
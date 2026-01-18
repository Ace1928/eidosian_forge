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
def testFlattenDictOrder(self):
    ordered = collections.OrderedDict([('d', 3), ('b', 1), ('a', 0), ('c', 2)])
    plain = {'d': 3, 'b': 1, 'a': 0, 'c': 2}
    ordered_flat = tree.flatten(ordered)
    plain_flat = tree.flatten(plain)
    self.assertEqual([0, 1, 2, 3], ordered_flat)
    self.assertEqual([0, 1, 2, 3], plain_flat)
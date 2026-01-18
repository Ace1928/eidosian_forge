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
def testUnflattenDictOrder(self):
    ordered = collections.OrderedDict([('d', 0), ('b', 0), ('a', 0), ('c', 0)])
    plain = {'d': 0, 'b': 0, 'a': 0, 'c': 0}
    seq = [0, 1, 2, 3]
    ordered_reconstruction = tree.unflatten_as(ordered, seq)
    plain_reconstruction = tree.unflatten_as(plain, seq)
    self.assertEqual(collections.OrderedDict([('d', 3), ('b', 1), ('a', 0), ('c', 2)]), ordered_reconstruction)
    self.assertEqual({'d': 3, 'b': 1, 'a': 0, 'c': 2}, plain_reconstruction)
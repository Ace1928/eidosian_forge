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
def testFlattenAndUnflatten(self):
    structure = ((3, 4), 5, (6, 7, (9, 10), 8))
    flat = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    self.assertEqual(tree.flatten(structure), [3, 4, 5, 6, 7, 9, 10, 8])
    self.assertEqual(tree.unflatten_as(structure, flat), (('a', 'b'), 'c', ('d', 'e', ('f', 'g'), 'h')))
    point = collections.namedtuple('Point', ['x', 'y'])
    structure = (point(x=4, y=2), ((point(x=1, y=0),),))
    flat = [4, 2, 1, 0]
    self.assertEqual(tree.flatten(structure), flat)
    restructured_from_flat = tree.unflatten_as(structure, flat)
    self.assertEqual(restructured_from_flat, structure)
    self.assertEqual(restructured_from_flat[0].x, 4)
    self.assertEqual(restructured_from_flat[0].y, 2)
    self.assertEqual(restructured_from_flat[1][0][0].x, 1)
    self.assertEqual(restructured_from_flat[1][0][0].y, 0)
    self.assertEqual([5], tree.flatten(5))
    self.assertEqual([np.array([5])], tree.flatten(np.array([5])))
    self.assertEqual('a', tree.unflatten_as(5, ['a']))
    self.assertEqual(np.array([5]), tree.unflatten_as('scalar', [np.array([5])]))
    with self.assertRaisesRegex(ValueError, 'Structure is a scalar'):
        tree.unflatten_as('scalar', [4, 5])
    with self.assertRaisesRegex(TypeError, 'flat_sequence'):
        tree.unflatten_as([4, 5], 'bad_sequence')
    with self.assertRaises(ValueError):
        tree.unflatten_as([5, 6, [7, 8]], ['a', 'b', 'c'])
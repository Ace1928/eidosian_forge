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
@parameterized.named_parameters([dict(testcase_name='Tuples', s1=(1, 2), s2=(3, 4), check_types=True, expected=(((0,), 4), ((1,), 6))), dict(testcase_name='Dicts', s1={'a': 1, 'b': 2}, s2={'b': 4, 'a': 3}, check_types=True, expected={'a': (('a',), 4), 'b': (('b',), 6)}), dict(testcase_name='Mixed', s1=(1, 2), s2=[3, 4], check_types=False, expected=(((0,), 4), ((1,), 6))), dict(testcase_name='Nested', s1={'a': [2, 3], 'b': [1, 2, 3]}, s2={'b': [5, 6, 7], 'a': [8, 9]}, check_types=True, expected={'a': [(('a', 0), 10), (('a', 1), 12)], 'b': [(('b', 0), 6), (('b', 1), 8), (('b', 2), 10)]})])
def testMapWithPathCompatibleStructures(self, s1, s2, check_types, expected):

    def path_and_sum(path, *values):
        return (path, sum(values))
    result = tree.map_structure_with_path(path_and_sum, s1, s2, check_types=check_types)
    self.assertEqual(expected, result)
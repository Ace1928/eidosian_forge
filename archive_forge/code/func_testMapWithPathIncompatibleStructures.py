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
@parameterized.named_parameters([dict(testcase_name='Tuples', s1=(1, 2, 3), s2=(4, 5), error_type=ValueError), dict(testcase_name='Dicts', s1={'a': 1}, s2={'b': 2}, error_type=ValueError), dict(testcase_name='Nested', s1={'a': [2, 3, 4], 'b': [1, 3]}, s2={'b': [5, 6], 'a': [8, 9]}, error_type=ValueError)])
def testMapWithPathIncompatibleStructures(self, s1, s2, error_type):
    with self.assertRaises(error_type):
        tree.map_structure_with_path(lambda path, *s: 0, s1, s2)
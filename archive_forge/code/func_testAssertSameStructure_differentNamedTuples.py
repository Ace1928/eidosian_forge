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
def testAssertSameStructure_differentNamedTuples(self):
    self.assertRaises(TypeError, tree.assert_same_structure, NestTest.Named0ab(3, 4), NestTest.Named1ab(3, 4))
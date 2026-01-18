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
def testNoneNodeIncluded(self):
    structure = (1, None)
    self.assertEqual(tree.flatten(structure), [1, None])
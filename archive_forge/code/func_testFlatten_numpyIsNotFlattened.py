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
def testFlatten_numpyIsNotFlattened(self):
    structure = np.array([1, 2, 3])
    flattened = tree.flatten(structure)
    self.assertLen(flattened, 1)
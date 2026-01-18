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
def testFlatten_stringIsNotFlattened(self):
    structure = 'lots of letters'
    flattened = tree.flatten(structure)
    self.assertLen(flattened, 1)
    self.assertEqual(structure, tree.unflatten_as('goodbye', flattened))
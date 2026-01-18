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
def testFlatten_bytearrayIsNotFlattened(self):
    structure = bytearray('bytes in an array', 'ascii')
    flattened = tree.flatten(structure)
    self.assertLen(flattened, 1)
    self.assertEqual(flattened, [structure])
    self.assertEqual(structure, tree.unflatten_as(bytearray('hello', 'ascii'), flattened))
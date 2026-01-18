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
def testAssertSameStructure_sameNamedTupleDifferentStructuredContents(self):
    with self.assertRaisesRegex(ValueError, "don't have the same nested structure\\.\n\nFirst structure: .*?\n\nSecond structure: "):
        tree.assert_same_structure(NestTest.Named0ab(3, 4), NestTest.Named0ab([3], 4))
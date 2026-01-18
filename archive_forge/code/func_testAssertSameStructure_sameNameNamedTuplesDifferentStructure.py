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
def testAssertSameStructure_sameNameNamedTuplesDifferentStructure(self):
    expected_message = "The two structures don't have the same.*"
    with self.assertRaisesRegex(ValueError, expected_message):
        tree.assert_same_structure(NestTest.SameNameab(0, NestTest.SameNameab2(1, 2)), NestTest.SameNameab2(NestTest.SameNameab(0, 1), 2))
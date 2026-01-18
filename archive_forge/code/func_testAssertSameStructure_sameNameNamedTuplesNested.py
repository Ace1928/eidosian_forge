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
def testAssertSameStructure_sameNameNamedTuplesNested(self):
    tree.assert_same_structure(NestTest.SameNameab(NestTest.SameName1xy(0, 1), 2), NestTest.SameNameab2(NestTest.SameName1xy2(2, 3), 4))
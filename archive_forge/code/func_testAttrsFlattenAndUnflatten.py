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
def testAttrsFlattenAndUnflatten(self):

    class BadAttr(object):
        """Class that has a non-iterable __attrs_attrs__."""
        __attrs_attrs__ = None

    @attr.s
    class SampleAttr(object):
        field1 = attr.ib()
        field2 = attr.ib()
    field_values = [1, 2]
    sample_attr = SampleAttr(*field_values)
    self.assertFalse(tree._is_attrs(field_values))
    self.assertTrue(tree._is_attrs(sample_attr))
    flat = tree.flatten(sample_attr)
    self.assertEqual(field_values, flat)
    restructured_from_flat = tree.unflatten_as(sample_attr, flat)
    self.assertIsInstance(restructured_from_flat, SampleAttr)
    self.assertEqual(restructured_from_flat, sample_attr)
    with self.assertRaisesRegex(TypeError, 'object is not iterable'):
        flat = tree.flatten(BadAttr())
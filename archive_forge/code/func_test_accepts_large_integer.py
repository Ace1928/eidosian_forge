import unittest
from traits.api import BaseFloat, Either, Float, HasTraits, Str, TraitError
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_accepts_large_integer(self):
    a = self.test_class()
    a.value = 2 ** 64
    self.assertIs(type(a.value), float)
    self.assertEqual(a.value, 2 ** 64)
    a.value_or_none = 2 ** 64
    self.assertIs(type(a.value_or_none), float)
    self.assertEqual(a.value_or_none, 2 ** 64)
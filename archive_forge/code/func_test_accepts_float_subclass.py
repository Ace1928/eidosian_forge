import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_accepts_float_subclass(self):
    self.model.percentage = InheritsFromFloat(44.0)
    self.assertIs(type(self.model.percentage), float)
    self.assertEqual(self.model.percentage, 44.0)
    with self.assertRaises(TraitError):
        self.model.percentage = InheritsFromFloat(-0.5)
    with self.assertRaises(TraitError):
        self.model.percentage = InheritsFromFloat(100.5)
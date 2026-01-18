import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_accepts_int_subclass(self):
    self.model.percentage = InheritsFromInt(44)
    self.assertIs(type(self.model.percentage), int)
    self.assertEqual(self.model.percentage, 44)
    with self.assertRaises(TraitError):
        self.model.percentage = InheritsFromInt(-1)
    with self.assertRaises(TraitError):
        self.model.percentage = InheritsFromInt(101)
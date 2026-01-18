import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_accepts_float_like(self):
    self.model.percentage = FloatLike(35.0)
    self.assertIs(type(self.model.percentage), float)
    self.assertEqual(self.model.percentage, 35.0)
    with self.assertRaises(TraitError):
        self.model.percentage = FloatLike(-0.5)
    with self.assertRaises(TraitError):
        self.model.percentage = FloatLike(100.5)
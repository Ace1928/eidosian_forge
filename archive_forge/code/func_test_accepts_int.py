import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_accepts_int(self):
    self.model.percentage = 35
    self.assertIs(type(self.model.percentage), float)
    self.assertEqual(self.model.percentage, 35.0)
    with self.assertRaises(TraitError):
        self.model.percentage = -1
    with self.assertRaises(TraitError):
        self.model.percentage = 101
import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_accepts_bool(self):
    self.model.percentage = False
    self.assertIs(type(self.model.percentage), float)
    self.assertEqual(self.model.percentage, 0.0)
    self.model.percentage = True
    self.assertIs(type(self.model.percentage), float)
    self.assertEqual(self.model.percentage, 1.0)
import unittest
from traits.api import (
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_bad_float_like(self):
    with self.assertRaises(ZeroDivisionError):
        self.model.percentage = BadFloatLike()
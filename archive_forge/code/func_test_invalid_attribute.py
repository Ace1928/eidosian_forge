from unittest import TestCase
from fastimport.helpers import (
from fastimport import (
def test_invalid_attribute(self):
    self.assertRaises(TypeError, self.c.copy, invalid=True)
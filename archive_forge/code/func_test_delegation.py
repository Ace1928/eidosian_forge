import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_delegation(self):
    obj = DelegateTrait3()
    self.assertEqual(obj.value, 99.0)
    parent1 = obj.delegate
    parent2 = parent1.delegate
    parent3 = parent2.delegate
    parent3.value = 3.0
    self.assertEqual(obj.value, 3.0)
    parent2.value = 2.0
    self.assertEqual(obj.value, 2.0)
    self.assertEqual(parent3.value, 3.0)
    parent1.value = 1.0
    self.assertEqual(obj.value, 1.0)
    self.assertEqual(parent2.value, 2.0)
    self.assertEqual(parent3.value, 3.0)
    obj.value = 0.0
    self.assertEqual(obj.value, 0.0)
    self.assertEqual(parent1.value, 1.0)
    self.assertEqual(parent2.value, 2.0)
    self.assertEqual(parent3.value, 3.0)
    del obj.value
    self.assertEqual(obj.value, 1.0)
    del parent1.value
    self.assertEqual(obj.value, 2.0)
    self.assertEqual(parent1.value, 2.0)
    del parent2.value
    self.assertEqual(obj.value, 3.0)
    self.assertEqual(parent1.value, 3.0)
    self.assertEqual(parent2.value, 3.0)
    del parent3.value
    self.assertEqual(obj.value, 99.0)
    self.assertEqual(parent1.value, 99.0)
    self.assertEqual(parent2.value, 99.0)
    self.assertEqual(parent3.value, 99.0)
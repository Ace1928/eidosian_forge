import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_assign_post_setattr_none(self):
    old_value = 'old_value'
    new_value = 'new_value'

    def post_setattr_func(obj, name, value):
        obj.output_variable = value
    trait = CTrait(TraitKind.trait)

    class TestClass(HasTraits):
        atr = trait
    trait.post_setattr = post_setattr_func
    self.assertIsNotNone(trait.post_setattr)
    obj = TestClass()
    obj.atr = old_value
    self.assertEqual(old_value, obj.output_variable)
    trait.post_setattr = None
    self.assertIsNone(trait.post_setattr)
    obj.atr = new_value
    self.assertEqual(old_value, obj.output_variable)
    trait.post_setattr = post_setattr_func
    obj.atr = old_value
    self.assertEqual(old_value, obj.output_variable)
    with self.assertRaises(ValueError):
        trait.post_setattr = 'Invalid'
import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_is_property(self):
    trait = CTrait(TraitKind.trait)
    self.assertFalse(trait.is_property)
    trait.property_fields = (getter, setter, validator)
    self.assertTrue(trait.is_property)
    with self.assertRaises(AttributeError):
        trait.is_property = False
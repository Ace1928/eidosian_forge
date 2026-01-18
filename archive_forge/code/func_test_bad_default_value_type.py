import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_bad_default_value_type(self):
    trait = CTrait(TraitKind.trait)
    with self.assertRaises(ValueError):
        trait.set_default_value(-1, None)
    with self.assertRaises(ValueError):
        trait.set_default_value(MAXIMUM_DEFAULT_VALUE_TYPE + 1, None)
import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_is_mapped(self):
    trait = CTrait(TraitKind.trait)
    self.assertFalse(trait.is_mapped)
    trait.is_mapped = True
    self.assertTrue(trait.is_mapped)
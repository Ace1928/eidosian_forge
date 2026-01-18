import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_invalid_comparison_mode(self):
    trait = CTrait(TraitKind.trait)
    with self.assertRaises(ValueError):
        trait.comparison_mode = -1
    with self.assertRaises(ValueError):
        trait.comparison_mode = 3
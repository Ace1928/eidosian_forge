import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_comparison_mode_unchanged_if_invalid(self):
    trait = CTrait(TraitKind.trait)
    default_comparison_mode = trait.comparison_mode
    self.assertNotEqual(default_comparison_mode, ComparisonMode.none)
    trait.comparison_mode = ComparisonMode.none
    with self.assertRaises(ValueError):
        trait.comparison_mode = -1
    self.assertEqual(trait.comparison_mode, ComparisonMode.none)
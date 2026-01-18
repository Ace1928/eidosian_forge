import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_initial_default_value(self):
    trait = CTrait(TraitKind.trait)
    self.assertEqual(trait.default_value(), (DefaultValue.constant, None))
import gc
import sys
import unittest
from traits.constants import DefaultValue
from traits.has_traits import (
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_errors import TraitError
from traits.trait_type import TraitType
from traits.trait_types import (
def test_default_value_for_no_cache(self):
    """ Make sure that CTrait.default_value_for() does not cache the
        result.
        """
    dummy = Dummy()
    self.assertEqual(dummy.__dict__, {})
    ctrait = dummy.trait('x')
    default = ctrait.default_value_for(dummy, 'x')
    self.assertEqual(default, 10)
    self.assertEqual(dummy.__dict__, {})
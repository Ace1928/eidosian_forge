import sys
import unittest.mock
import warnings
import weakref
from traits.api import HasTraits
from traits.constants import (
from traits.ctrait import CTrait
from traits.trait_errors import TraitError
from traits.trait_types import Any, Int, List
def test_default_value_for_set_is_deprecated(self):
    trait = CTrait(TraitKind.trait)
    with warnings.catch_warnings(record=True) as warn_msgs:
        warnings.simplefilter('always', DeprecationWarning)
        trait.default_value(DefaultValue.constant, 3.7)
    self.assertEqual(len(warn_msgs), 1)
    warn_msg = warn_msgs[0]
    self.assertIn('default_value method with arguments is deprecated', str(warn_msg.message))
    _, _, this_module = __name__.rpartition('.')
    self.assertIn(this_module, warn_msg.filename)
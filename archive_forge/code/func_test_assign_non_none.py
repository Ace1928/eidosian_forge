import unittest
from traits.api import HasTraits, TraitError
from traits.trait_types import _NoneTrait
def test_assign_non_none(self):
    with self.assertRaises(TraitError):
        A(none_atr=5)
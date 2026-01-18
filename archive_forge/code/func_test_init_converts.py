import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_init_converts(self):
    tl = TraitList([True, False], item_validator=int_item_validator)
    self.assertEqual(tl, [1, 0])
    self.assertTrue(all((type(item) is int for item in tl)), msg='Non-integers found in int-only list')
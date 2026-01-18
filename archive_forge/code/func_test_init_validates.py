import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_init_validates(self):
    with self.assertRaises(TraitError):
        TraitList([1, 2.0, 3], item_validator=int_item_validator)
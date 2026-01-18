import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_imul_does_not_revalidate(self):
    item_validator = unittest.mock.Mock(wraps=int_item_validator)
    tl = TraitList([1, 1], item_validator=item_validator)
    item_validator.reset_mock()
    tl *= 3
    item_validator.assert_not_called()
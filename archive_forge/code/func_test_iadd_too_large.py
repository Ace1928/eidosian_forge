import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_iadd_too_large(self):
    foo = HasLengthConstrainedLists(at_most_five=[1, 2, 3, 4])
    with self.assertRaises(TraitError):
        foo.at_most_five += [6, 7, 8]
    self.assertEqual(foo.at_most_five, [1, 2, 3, 4])
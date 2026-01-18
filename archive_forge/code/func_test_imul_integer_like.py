import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
@requires_numpy
def test_imul_integer_like(self):
    tl = TraitList([1, 2], item_validator=int_item_validator, notifiers=[self.notification_handler])
    tl *= numpy.int64(2)
    self.assertEqual(tl, [1, 2, 1, 2])
    self.assertEqual(self.index, 2)
    self.assertEqual(self.removed, [])
    self.assertEqual(self.added, [1, 2])
    tl *= numpy.int64(-1)
    self.assertEqual(tl, [])
    self.assertEqual(self.index, 0)
    self.assertEqual(self.removed, [1, 2, 1, 2])
    self.assertEqual(self.added, [])
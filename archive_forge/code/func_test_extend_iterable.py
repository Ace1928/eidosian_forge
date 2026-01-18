import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_extend_iterable(self):
    tl = TraitList([1], item_validator=int_item_validator, notifiers=[self.notification_handler])
    tl.extend((x ** 2 for x in range(10, 13)))
    self.assertEqual(tl, [1, 100, 121, 144])
    self.assertEqual(self.index, 1)
    self.assertEqual(self.removed, [])
    self.assertEqual(self.added, [100, 121, 144])
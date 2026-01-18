import copy
import operator
import pickle
import unittest.mock
from traits.api import HasTraits, Int, List
from traits.testing.optional_dependencies import numpy, requires_numpy
from traits.trait_base import _validate_everything
from traits.trait_errors import TraitError
from traits.trait_list_object import (
def test_append_converts(self):
    tl = TraitList([2], item_validator=int_item_validator, notifiers=[self.notification_handler])
    tl.append(False)
    self.assertEqual(tl, [2, 0])
    self.assertEqual(self.index, 1)
    self.assertEqual(self.removed, [])
    self.assertEqual(self.added, [0])
    self.assertTrue(all((type(item) is int for item in tl)), msg='Non-integers found in int-only list')
    self.assertTrue(all((type(item) is int for item in self.added)), msg='Event contains non-integers for int-only list')
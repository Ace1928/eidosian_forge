import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_list2(self):
    self.check_list(self.obj.list2)
    self.assertRaises(TraitError, self.del_range, self.obj.list2, 0, 1)
    self.assertRaises(TraitError, self.del_extended_slice, self.obj.list2, 4, -5, -1)
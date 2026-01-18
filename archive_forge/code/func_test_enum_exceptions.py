import unittest
import warnings
from traits.api import (
from traits.testing.optional_dependencies import requires_traitsui
def test_enum_exceptions(self):
    """ Check that enumerated values can be combined with nested
        TraitCompound handlers.
        """
    self.check_values('num4', 1, [1, 2, 3, -3, -2, -1, 10, ()], [0, 4, 5, -5, -4, 11])
    self.check_values('num5', 1, [1, 2, 3, -3, -2, -1, 10, ()], [0, 4, 5, -5, -4, 11])
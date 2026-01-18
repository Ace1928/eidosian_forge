from traits.api import HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
def test_invalid_assignment_length(self):
    self._assign_invalid_values_length(('str', 44))
    self._assign_invalid_values_length(('str', 33, None, []))
import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import HasStrictTraits, Time, TraitError
def test_assign_none_with_allow_none_true(self):
    obj = HasTimeTraits(none_allowed=UNIX_EPOCH)
    self.assertIsNotNone(obj.none_allowed)
    obj.none_allowed = None
    self.assertIsNone(obj.none_allowed)
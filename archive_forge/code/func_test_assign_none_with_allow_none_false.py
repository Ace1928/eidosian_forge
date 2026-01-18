import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import HasStrictTraits, Time, TraitError
def test_assign_none_with_allow_none_false(self):
    obj = HasTimeTraits(none_prohibited=UNIX_EPOCH)
    with self.assertRaises(TraitError) as exception_context:
        obj.none_prohibited = None
    message = str(exception_context.exception)
    self.assertIn('must be a time, but', message)
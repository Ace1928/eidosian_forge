import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import HasStrictTraits, Time, TraitError
def test_assign_datetime(self):
    obj = HasTimeTraits()
    with self.assertRaises(TraitError) as exception_context:
        obj.simple_time = datetime.datetime(1975, 2, 13)
    message = str(exception_context.exception)
    self.assertIn('must be a time or None, but', message)
    self.assertIsNone(obj.simple_time)
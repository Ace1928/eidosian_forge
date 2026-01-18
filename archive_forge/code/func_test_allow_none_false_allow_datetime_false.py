import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import Date, HasStrictTraits, TraitError
def test_allow_none_false_allow_datetime_false(self):
    obj = HasDateTraits(strict=UNIX_EPOCH)
    with self.assertRaises(TraitError) as exception_context:
        obj.strict = None
    message = str(exception_context.exception)
    self.assertIn('must be a non-datetime date, but', message)
import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import Date, HasStrictTraits, TraitError
def test_assign_datetime_with_allow_datetime_not_given(self):
    test_datetime = datetime.datetime(1975, 2, 13)
    obj = HasDateTraits()
    with self.assertWarns(DeprecationWarning) as warnings_cm:
        obj.simple_date = test_datetime
    self.assertEqual(obj.simple_date, test_datetime)
    _, _, this_module = __name__.rpartition('.')
    self.assertIn(this_module, warnings_cm.filename)
    self.assertIn('datetime instances will no longer be accepted', str(warnings_cm.warning))
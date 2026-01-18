import datetime
import unittest
from traits.testing.optional_dependencies import requires_traitsui, traitsui
from traits.api import HasStrictTraits, Time, TraitError
def test_assign_none_with_allow_none_not_given(self):
    obj = HasTimeTraits(simple_time=UNIX_EPOCH)
    self.assertIsNotNone(obj.simple_time)
    with self.assertWarns(DeprecationWarning) as warnings_cm:
        obj.simple_time = None
    self.assertIsNone(obj.simple_time)
    _, _, this_module = __name__.rpartition('.')
    self.assertIn(this_module, warnings_cm.filename)
    self.assertIn('None will no longer be accepted', str(warnings_cm.warning))
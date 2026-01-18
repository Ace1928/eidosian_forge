import unittest
from traits.api import HasTraits, Int, List, Range, Str, TraitError, Tuple
def test_dynamic_events(self):
    obj = WithDynamicRange()
    obj._changed_handler_calls = 0
    obj.r = 5
    self.assertEqual(obj._changed_handler_calls, 1)
    self.assertEqual(obj.r, 5)
    with self.assertRaises(TraitError):
        obj.r = obj.high
    self.assertEqual(obj.r, 5)
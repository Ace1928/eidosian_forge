import unittest
from traits.api import HasTraits, Int, List, Range, Str, TraitError, Tuple
def test_non_ui_events(self):
    obj = WithFloatRange()
    obj._changed_handler_calls = 0
    obj.r = 10
    self.assertEqual(1, obj._changed_handler_calls)
    obj._changed_handler_calls = 0
    obj.r = 34.56
    self.assertEqual(obj._changed_handler_calls, 2)
    self.assertEqual(obj.r, 40)
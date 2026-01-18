import unittest
from traits.api import HasTraits, Int, List, Range, Str, TraitError, Tuple
def test_non_ui_int_events(self):
    obj = WithLargeIntRange()
    obj._changed_handler_calls = 0
    obj.r = 10
    self.assertEqual(obj._changed_handler_calls, 1)
    self.assertEqual(obj.r, 10)
    obj.r = 100
    self.assertEqual(obj._changed_handler_calls, 2)
    self.assertEqual(obj.r, 100)
    obj.r = 101
    self.assertEqual(obj._changed_handler_calls, 4)
    self.assertEqual(obj.r, 0)
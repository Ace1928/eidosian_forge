import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def test_extended_notifiers_methods(self):
    obj = ExtendedNotifiers(ok=2)
    obj.ok = 3
    self.assertEqual(len(obj.rebind_calls_0), 2)
    expected_1 = [2, 3]
    self.assertEqual(expected_1, obj.rebind_calls_1)
    expected_2 = [('ok', 2), ('ok', 3)]
    self.assertEqual(expected_2, obj.rebind_calls_2)
    expected_3 = [(obj, 'ok', 2), (obj, 'ok', 3)]
    self.assertEqual(expected_3, obj.rebind_calls_3)
    expected_4 = [(obj, 'ok', 0, 2), (obj, 'ok', 2, 3)]
    self.assertEqual(expected_4, obj.rebind_calls_4)
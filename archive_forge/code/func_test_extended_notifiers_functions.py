import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def test_extended_notifiers_functions(self):
    calls_0.clear()
    calls_1.clear()
    calls_2.clear()
    calls_3.clear()
    calls_4.clear()
    obj = ExtendedNotifiers()
    obj._on_trait_change(function_listener_0, 'ok', dispatch='extended')
    obj._on_trait_change(function_listener_1, 'ok', dispatch='extended')
    obj._on_trait_change(function_listener_2, 'ok', dispatch='extended')
    obj._on_trait_change(function_listener_3, 'ok', dispatch='extended')
    obj._on_trait_change(function_listener_4, 'ok', dispatch='extended')
    obj.ok = 2
    obj.ok = 3
    expected_0 = [True, True]
    self.assertEqual(expected_0, calls_0)
    expected_1 = [2, 3]
    self.assertEqual(expected_1, calls_1)
    expected_2 = [('ok', 2), ('ok', 3)]
    self.assertEqual(expected_2, calls_2)
    expected_3 = [(obj, 'ok', 2), (obj, 'ok', 3)]
    self.assertEqual(expected_3, calls_3)
    expected_4 = [(obj, 'ok', 0, 2), (obj, 'ok', 2, 3)]
    self.assertEqual(expected_4, calls_4)
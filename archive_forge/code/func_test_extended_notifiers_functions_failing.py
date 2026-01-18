import unittest
from traits import trait_notifiers
from traits.api import Float, HasTraits, List
def test_extended_notifiers_functions_failing(self):
    obj = ExtendedNotifiers()
    exceptions_from.clear()
    obj._on_trait_change(failing_function_listener_0, 'fail', dispatch='extended')
    obj._on_trait_change(failing_function_listener_1, 'fail', dispatch='extended')
    obj._on_trait_change(failing_function_listener_2, 'fail', dispatch='extended')
    obj._on_trait_change(failing_function_listener_3, 'fail', dispatch='extended')
    obj._on_trait_change(failing_function_listener_4, 'fail', dispatch='extended')
    obj.fail = 1
    self.assertCountEqual([0, 1, 2, 3, 4], obj.exceptions_from)
    self.assertEqual([(obj, 'fail', 0, 1)] * 10, self.exceptions)
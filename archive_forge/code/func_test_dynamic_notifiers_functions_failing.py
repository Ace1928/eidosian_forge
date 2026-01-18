import gc
import unittest
from traits import trait_notifiers
from traits.api import Event, Float, HasTraits, List, on_trait_change
def test_dynamic_notifiers_functions_failing(self):
    obj = DynamicNotifiers()
    exceptions_from = []

    def failing_function_listener_0():
        exceptions_from.append(0)
        raise Exception('error')

    def failing_function_listener_1(new):
        exceptions_from.append(1)
        raise Exception('error')

    def failing_function_listener_2(name, new):
        exceptions_from.append(2)
        raise Exception('error')

    def failing_function_listener_3(obj, name, new):
        exceptions_from.append(3)
        raise Exception('error')

    def failing_function_listener_4(obj, name, old, new):
        exceptions_from.append(4)
        raise Exception('error')
    obj.on_trait_change(failing_function_listener_0, 'fail')
    obj.on_trait_change(failing_function_listener_1, 'fail')
    obj.on_trait_change(failing_function_listener_2, 'fail')
    obj.on_trait_change(failing_function_listener_3, 'fail')
    obj.on_trait_change(failing_function_listener_4, 'fail')
    obj.fail = 1
    self.assertEqual([0, 1, 2, 3, 4], exceptions_from)
    self.assertCountEqual([0, 1, 2, 3, 4], obj.exceptions_from)
    self.assertEqual([(obj, 'fail', 0, 1)] * 10, self.exceptions)
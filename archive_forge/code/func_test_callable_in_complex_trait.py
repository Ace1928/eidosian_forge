import inspect
import unittest
from traits.api import (
def test_callable_in_complex_trait(self):
    a = MyCallable()
    self.assertIsNone(a.callable_or_str)
    acceptable_values = [pow, 'pow', None, int]
    for value in acceptable_values:
        a.callable_or_str = value
        self.assertEqual(a.callable_or_str, value)
    unacceptable_values = [1.0, 3j, (5, 6, 7)]
    for value in unacceptable_values:
        old_value = a.callable_or_str
        with self.assertRaises(TraitError):
            a.callable_or_str = value
        self.assertEqual(a.callable_or_str, old_value)
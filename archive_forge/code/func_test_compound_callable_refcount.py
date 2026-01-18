import inspect
import unittest
from traits.api import (
def test_compound_callable_refcount(self):

    def my_function():
        return 72
    a = MyCallable()
    string_value = 'some string'
    callable_value = my_function
    for _ in range(5):
        a.callable_or_str = string_value
        a.callable_or_str = callable_value
    self.assertEqual(a.callable_or_str(), 72)
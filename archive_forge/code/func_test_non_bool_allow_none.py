import inspect
import unittest
from traits.api import (
def test_non_bool_allow_none(self):
    obj = MyCallable()
    obj.bad_allow_none = 'a string'
    with self.assertRaises(ZeroDivisionError):
        obj.bad_allow_none = None
import inspect
import unittest
from traits.api import (
def test_implicitly_allowed_none_in_compound(self):
    obj = MyCallable()
    obj.old_callable_or_str = 'bob'
    obj.old_callable_or_str = None
    self.assertIsNone(obj.old_callable_or_str)
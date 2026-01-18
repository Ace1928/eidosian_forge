import inspect
import unittest
from traits.api import (
def test_disallow_none(self):

    class MyNewCallable(HasTraits):
        value = Callable(default_value=pow, allow_none=False)
    obj = MyNewCallable()
    self.assertIsNotNone(obj.value)
    with self.assertRaises(TraitError):
        obj.value = None
    self.assertEqual(8, obj.value(2, 3))
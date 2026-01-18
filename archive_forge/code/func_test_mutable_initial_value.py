import unittest
from traits.api import Constant, HasTraits, TraitError
def test_mutable_initial_value(self):

    class TestClass(HasTraits):
        c_atr_1 = Constant([1, 2, 3, 4, 5])
        c_atr_2 = Constant({'a': 1, 'b': 2})
    obj = TestClass()
    self.assertEqual(obj.c_atr_1, [1, 2, 3, 4, 5])
    self.assertEqual(obj.c_atr_2, {'a': 1, 'b': 2})
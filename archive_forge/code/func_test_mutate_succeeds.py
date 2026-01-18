import unittest
from traits.api import Constant, HasTraits, TraitError
def test_mutate_succeeds(self):

    class TestClass(HasTraits):
        c_atr_1 = Constant([1, 2, 3, 4, 5])
        c_atr_2 = Constant({'a': 1, 'b': 2})
    obj = TestClass()
    obj.c_atr_1.append(6)
    obj.c_atr_2['c'] = 3
    self.assertEqual(obj.c_atr_1, [1, 2, 3, 4, 5, 6])
    self.assertEqual(obj.c_atr_2, {'a': 1, 'b': 2, 'c': 3})
import unittest
from traits.api import Constant, HasTraits, TraitError
def test_mutate_affects_all_instances(self):

    class TestClass(HasTraits):
        c_atr = Constant([1, 2, 3, 4, 5])
    obj1 = TestClass()
    obj2 = TestClass()
    obj2.c_atr.append(6)
    self.assertEqual(obj1.c_atr, [1, 2, 3, 4, 5, 6])
    self.assertIs(obj1.c_atr, obj2.c_atr)
import abc
import unittest
import warnings
from traits.api import ABCHasTraits, ABCMetaHasTraits, HasTraits, Int, Float
def test_basic_abc(self):
    self.assertRaises(TypeError, AbstractFoo)
    concrete = ConcreteFoo()
    self.assertEqual(concrete.foo(), 'foo')
    self.assertEqual(concrete.bar, 'bar')
    self.assertEqual(concrete.x, 10)
    self.assertEqual(concrete.y, 20.0)
    self.assertTrue(isinstance(concrete, AbstractFoo))
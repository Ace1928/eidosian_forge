import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
def test_default_method(self):

    class Person(HasTraits):
        married = PrefixMap({'yes': 1, 'yeah': 1, 'no': 0, 'nah': 0})
        default_calls = Int(0)

        def _married_default(self):
            self.default_calls += 1
            return 'nah'
    p = Person()
    self.assertEqual(p.married, 'nah')
    self.assertEqual(p.married_, 0)
    self.assertEqual(p.default_calls, 1)
    p2 = Person()
    self.assertEqual(p2.married_, 0)
    self.assertEqual(p2.married, 'nah')
    self.assertEqual(p2.default_calls, 1)
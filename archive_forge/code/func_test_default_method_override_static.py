import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
def test_default_method_override_static(self):

    class BasePerson(HasTraits):
        married = PrefixMap({'yes': 1, 'yeah': 1, 'no': 0, 'nah': 0})
        default_calls = Int(0)

        def _married_default(self):
            self.default_calls += 1
            return 'nah'

    class Person(BasePerson):
        married = 'yes'
    p = Person()
    self.assertEqual(p.married, 'yes')
    self.assertEqual(p.married_, 1)
    self.assertEqual(p.default_calls, 0)
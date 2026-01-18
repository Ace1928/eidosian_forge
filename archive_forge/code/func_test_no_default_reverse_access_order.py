import pickle
import unittest
from traits.api import HasTraits, Int, List, Map, on_trait_change, TraitError
def test_no_default_reverse_access_order(self):
    mapping = {'yes': 1, 'yeah': 1, 'no': 0, 'nah': 0}

    class Person(HasTraits):
        married = Map(mapping)
    p = Person()
    shadow_value = p.married_
    primary_value = p.married
    self.assertEqual(primary_value, 'yes')
    self.assertEqual(shadow_value, 1)
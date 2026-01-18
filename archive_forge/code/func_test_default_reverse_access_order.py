import pickle
import unittest
from traits.api import HasTraits, Int, List, Map, on_trait_change, TraitError
def test_default_reverse_access_order(self):

    class Person(HasTraits):
        married = Map({'yes': 1, 'yeah': 1, 'no': 0, 'nah': 0, None: 2}, default_value=None)
    p = Person()
    self.assertEqual(p.married_, 2)
    self.assertIsNone(p.married)
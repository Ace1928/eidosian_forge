import pickle
import unittest
from traits.api import HasTraits, Int, PrefixMap, TraitError
def test_bad_types(self):
    person = Person()
    wrong_type = [[], (1, 2, 3), 1j, 2.3, 23, b'not a string', None]
    for value in wrong_type:
        with self.subTest(value=value):
            with self.assertRaises(TraitError):
                person.married = value
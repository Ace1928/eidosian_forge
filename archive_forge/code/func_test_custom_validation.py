import unittest
from traits.api import HasStrictTraits, Int, TraitError
from traits.tests.tuple_test_mixin import TupleTestMixin
from traits.trait_types import ValidatedTuple
def test_custom_validation(self):
    simple = Simple()
    simple.scalar_range = (2, 5)
    self.assertEqual(simple.scalar_range, (2, 5))
    with self.assertRaises(TraitError):
        simple.scalar_range = (5, 2)
import unittest
from traits.api import HasTraits, Int, List, Range, Str, TraitError, Tuple
def test_dynamic_range_in_tuple(self):

    class HasRangeInTuple(HasTraits):
        low = Int()
        high = Int()
        hours_and_name = Tuple(Range(low='low', high='high'), Str)
    model = HasRangeInTuple(low=0, high=48)
    model.hours_and_name = (40, 'fred')
    self.assertEqual(model.hours_and_name, (40, 'fred'))
    with self.assertRaises(TraitError):
        model.hours_and_name = (50, 'george')
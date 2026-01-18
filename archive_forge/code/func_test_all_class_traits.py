import unittest
from traits.api import HasTraits, Int, List, Str
def test_all_class_traits(self):
    expected = ['x', 'name', 'trait_added', 'trait_modified']
    self.assertCountEqual(A.class_traits(), expected)
    self.assertCountEqual(B.class_traits(), expected)
    expected.extend(('lst', 'y'))
    self.assertCountEqual(C.class_traits(), expected)
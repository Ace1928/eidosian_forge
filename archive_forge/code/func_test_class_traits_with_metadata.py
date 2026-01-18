import unittest
from traits.api import HasTraits, Int, List, Str
def test_class_traits_with_metadata(self):
    traits = C.class_traits(marked=True)
    self.assertCountEqual(list(traits.keys()), ('y', 'name'))
    marked_traits = C.class_traits(marked=lambda attr: attr is not None)
    self.assertCountEqual(marked_traits, ('y', 'name', 'lst'))
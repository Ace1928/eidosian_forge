import unittest
from traits.api import HasTraits, Instance, Str, Any, Property
def test_Any_circular_references(self):
    bar = BarAny()
    baz = BazAny()
    bar.other = baz
    baz.other = bar
    bar_copy = bar.clone_traits()
    self.assertIsNot(bar_copy, bar)
    self.assertIs(bar_copy.other, baz)
    self.assertIs(bar_copy.other.other, bar)
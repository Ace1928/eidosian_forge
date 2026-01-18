import unittest
from traits.api import ArrayOrNone, ComparisonMode, HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_comparison_mode_override(self):
    foo = Foo()
    test_array = numpy.arange(-7, 2)
    with self.assertTraitChanges(foo, 'maybe_array_no_compare'):
        foo.maybe_array_no_compare = None
    with self.assertTraitChanges(foo, 'maybe_array_no_compare'):
        foo.maybe_array_no_compare = test_array
    with self.assertTraitChanges(foo, 'maybe_array_no_compare'):
        foo.maybe_array_no_compare = test_array
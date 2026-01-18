import unittest
from traits.api import ArrayOrNone, ComparisonMode, HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_change_notifications(self):
    foo = Foo()
    test_array = numpy.arange(-7, -2)
    different_test_array = numpy.arange(10)
    with self.assertTraitDoesNotChange(foo, 'maybe_array'):
        foo.maybe_array = None
    with self.assertTraitChanges(foo, 'maybe_array'):
        foo.maybe_array = test_array
    with self.assertTraitDoesNotChange(foo, 'maybe_array'):
        foo.maybe_array = test_array
    with self.assertTraitChanges(foo, 'maybe_array'):
        foo.maybe_array = different_test_array
    different_test_array += 2
    with self.assertTraitDoesNotChange(foo, 'maybe_array'):
        foo.maybe_array = different_test_array
    with self.assertTraitChanges(foo, 'maybe_array'):
        foo.maybe_array = None
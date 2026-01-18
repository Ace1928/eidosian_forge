import unittest
from traits.api import ArrayOrNone, ComparisonMode, HasTraits, TraitError
from traits.testing.unittest_tools import UnittestTools
from traits.testing.optional_dependencies import numpy, requires_numpy
def test_default_value_copied(self):
    test_default = numpy.arange(100.0, 110.0)

    class FooBar(HasTraits):
        foo = ArrayOrNone(value=test_default)
        bar = ArrayOrNone(value=test_default)
    foo_bar = FooBar()
    self.assertTrue((foo_bar.foo == test_default).all())
    self.assertTrue((foo_bar.bar == test_default).all())
    test_default += 2.0
    self.assertFalse((foo_bar.foo == test_default).all())
    self.assertFalse((foo_bar.bar == test_default).all())
    foo = foo_bar.foo
    foo += 1729.0
    self.assertFalse((foo_bar.foo == foo_bar.bar).all())
import unittest
from traits.trait_types import Int
from traits.has_traits import HasTraits
def test_error_propagation_in_default_methods(self):

    class FooException(Foo):

        def _bar_default(self):
            1 / 0
    foo = FooException()
    self.assertRaises(ZeroDivisionError, lambda: foo.bar)

    class FooKeyError(Foo):

        def _bar_default(self):
            raise KeyError()
    foo = FooKeyError()
    self.assertRaises(KeyError, lambda: foo.bar)
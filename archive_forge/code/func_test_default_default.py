import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Any
def test_default_default(self):

    class A(HasTraits):
        foo = Any()
    a = A()
    self.assertEqual(a.foo, None)
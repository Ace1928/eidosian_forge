import unittest
from traits.api import (
from traits.tests.tuple_test_mixin import TupleTestMixin
def test_constant_defaults(self):

    class A(HasTraits):
        foo = Tuple(Int, Tuple(Str, Int))
    a = A()
    b = A()
    self.assertEqual(a.foo, (0, ('', 0)))
    self.assertIs(a.foo, b.foo)
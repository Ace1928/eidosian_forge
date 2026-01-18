import unittest
from traits.api import (
from traits.tests.tuple_test_mixin import TupleTestMixin
def test_non_constant_defaults(self):

    class A(HasTraits):
        foo = Tuple(List(Int))
    a = A()
    a.foo[0].append(35)
    self.assertEqual(a.foo[0], [35])
    with self.assertRaises(TraitError):
        a.foo[0].append(3.5)
    b = A()
    self.assertEqual(b.foo[0], [])
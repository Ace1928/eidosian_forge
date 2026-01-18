import unittest
from traits.has_traits import HasTraits
from traits.trait_types import Any
def test_dict_default(self):
    message_pattern = "a default value of type 'dict'.* will be shared"
    with self.assertWarnsRegex(DeprecationWarning, message_pattern):

        class A(HasTraits):
            foo = Any({})
    a = A()
    b = A()
    self.assertEqual(a.foo, {})
    self.assertEqual(b.foo, {})
    a.foo['color'] = 'red'
    self.assertEqual(a.foo, {'color': 'red'})
    self.assertEqual(b.foo, {})
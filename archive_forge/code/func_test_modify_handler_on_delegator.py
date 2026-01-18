import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def test_modify_handler_on_delegator(self):
    f = Foo()
    b = BazModify(foo=f)
    self.assertEqual(f.t, b.t)
    b.t = 'changed'
    self.assertEqual(f.t, b.t)
    self.assertEqual(baz_t_handler_self, b)
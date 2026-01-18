import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def test_no_modify_handler_on_delegator(self):
    f = Foo()
    b = BazNoModify(foo=f)
    self.assertEqual(f.t, b.t)
    b.t = 'changed'
    self.assertNotEqual(f.t, b.t)
    self.assertEqual(baz_t_handler_self, b)
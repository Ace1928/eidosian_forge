import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def test_no_modify_prefix_handler_on_delegatee_not_called(self):
    f = Foo()
    b = BazNoModify(foo=f)
    self.assertEqual(f.s, b.sd)
    b.sd = 'changed'
    self.assertNotEqual(f.s, b.sd)
    self.assertEqual(foo_s_handler_self, None)
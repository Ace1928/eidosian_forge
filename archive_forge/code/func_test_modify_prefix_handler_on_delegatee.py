import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def test_modify_prefix_handler_on_delegatee(self):
    f = Foo()
    b = BazModify(foo=f)
    self.assertEqual(f.s, b.sd)
    b.sd = 'changed'
    self.assertEqual(f.s, b.sd)
    self.assertEqual(foo_s_handler_self, f)
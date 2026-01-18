import unittest
from traits.api import Delegate, HasTraits, Instance, Str
def test_modify_prefix_handler_on_delegator(self):
    f = Foo()
    b = BazModify(foo=f)
    self.assertEqual(f.s, b.sd)
    b.sd = 'changed'
    self.assertEqual(f.s, b.sd)
    self.assertEqual(baz_s_handler_self, None)
    self.assertEqual(baz_sd_handler_self, b)
import unittest
from traits.api import HasTraits, Instance, Str
def test_baz2_bar_foo(self):
    self.assertIsNot(self.baz2.bar.foo, None)
    self.assertIsNot(self.baz2.bar.foo, self.foo2)
    self.assertIsNot(self.baz2.bar.foo, self.baz.bar.foo)
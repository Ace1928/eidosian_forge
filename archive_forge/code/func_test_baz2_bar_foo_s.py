import unittest
from traits.api import HasTraits, Instance, Str
def test_baz2_bar_foo_s(self):
    self.assertEqual(self.baz2.bar.foo.s, 'foo')
    self.assertEqual(self.baz2.bar.foo.s, self.baz.bar.foo.s)
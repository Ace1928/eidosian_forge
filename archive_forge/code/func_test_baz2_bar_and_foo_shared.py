import unittest
from traits.api import HasTraits, Instance, Str
def test_baz2_bar_and_foo_shared(self):
    self.assertIs(self.baz2.bar.shared, self.baz2.bar.foo.shared)
    self.assertIs(self.baz2.shared, self.baz2.bar.foo.shared)
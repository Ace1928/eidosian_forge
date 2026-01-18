import unittest
from traits.api import HasTraits, Instance, Str
def test_baz2_bar_shared(self):
    self.assertIsNot(self.baz2.bar.shared, None)
    self.assertIsNot(self.baz2.bar.shared, self.shared2)
    self.assertIs(self.baz2.bar.shared, self.shared)
    self.assertIs(self.baz2.bar.shared, self.baz2.shared)
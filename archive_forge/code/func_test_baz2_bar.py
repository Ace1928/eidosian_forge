import unittest
from traits.api import HasTraits, Instance, Str
def test_baz2_bar(self):
    self.assertIsNot(self.baz2.bar, None)
    self.assertIsNot(self.baz2.bar, self.bar2)
    self.assertIsNot(self.baz2.bar, self.baz.bar)